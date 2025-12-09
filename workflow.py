from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from crewai_tools import RagTool
from crewai.llms.providers.openai.completion import OpenAICompletion

from db import (
    DEFAULT_DB,
    apply_ingredient_patches,
    apply_nutrition_patches,
    apply_recipe_patches,
    fetch_tables,
    get_state,
    init_db,
    set_state,
    upsert_images,
    upsert_ingredients,
    upsert_nutrition,
    upsert_recipes,
)

# ---------------------------------------------------------------------
# Global image store: maps image_id -> raw bytes
# The LLM only ever sees small handles like {"image_id": "img_0", "name": "..."}
# ---------------------------------------------------------------------

IMAGE_STORE: Dict[str, bytes] = {}
# Simple counter for generated images
_GENERATED_IMAGE_COUNTER = 0

def register_generated_image(image_bytes: bytes, recipe_id: Optional[int], name: str) -> str:
    """
    Store generated image bytes in IMAGE_STORE and return a short handle string.

    The handle format is 'gen://<key>', where <key> can be used to look up the bytes in IMAGE_STORE.
    """
    global _GENERATED_IMAGE_COUNTER
    _GENERATED_IMAGE_COUNTER += 1
    safe_name = (name or "recipe").replace(" ", "_").replace("/", "_")
    key = f"gen_{recipe_id or 'new'}_{safe_name}_{_GENERATED_IMAGE_COUNTER}"
    IMAGE_STORE[key] = image_bytes
    return f"gen://{key}"

try:
    from openai import OpenAI

    openai_client: Optional["OpenAI"] = OpenAI()
except Exception:
    openai_client = None


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------

@tool("vision_detect_ingredients")
def vision_detect_ingredients_tool(payload: dict) -> str:
    """Tool wrapper around vision_detect_ingredients, returns JSON string."""
    result = vision_detect_ingredients(payload)
    return json.dumps(result)


# @tool("vision_detect_ingredients")
def vision_detect_ingredients(payload: dict) -> str:
    """
    Vision tool that turns uploaded images + hints into candidate ingredients.

    Input:
      payload: JSON with keys:
        - images: list of {image_id: str, name: str}
        - hints:  str

    Output:
      JSON with:
        - detected: list of ingredient rows
        - summary:  short natural language summary
    """
    if isinstance(payload, str):
        data = json.loads(payload)
    elif isinstance(payload, dict):
        data = payload
    else:
        raise ValueError("payload must be str or dict")

    # print("[VISION TOOL] called with payload:", payload)

    hints = data.get("hints") or ""
    image_handles = data.get("images") or []

    # # debugging
    # for img in image_handles:
    #     image_id = img.get("image_id")
    #     raw_bytes = IMAGE_STORE.get(image_id)
    #     print("[VISION TOOL] image_id:", image_id, "bytes:", len(raw_bytes) if raw_bytes else None)

    # If we don't have a live OpenAI client, just return stub detections
    if not openai_client:
        detected = []
        for img in image_handles:
            name = img.get("name", "image")
            detected.append(
                {
                    "name": f"ingredient_from_{name}",
                    "form": "unspecified",
                    "estimated_weight_g_or_ml": None,
                    "observed_quantity": None,
                    "spoiling_estimate_days_in_place": None,
                    "confidence": 0.25,
                }
            )
        return json.dumps(
            {
                "detected": detected,
                "summary": f"Stub vision tool run on {len(image_handles)} images. Hints: {hints}",
                "warning": "openai_client not available; used stub detections",
            }
        )

    # Real vision call using GPT-5.1 (or override via env)
    vision_model = os.getenv("OPENAI_VISION_MODEL", "gpt-5.1")

    # Build content blocks: one text + one image_url per handle
    image_blocks = []
    for img in image_handles:
        image_id = img.get("image_id")
        raw_bytes = IMAGE_STORE.get(image_id)
        if not raw_bytes:
            continue
        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"
        image_blocks.append({"type": "image_url", "image_url": {"url": data_url}})

    if not image_blocks:
        # Nothing usable; fall back to stub behavior
        detected = []
        for img in image_handles:
            name = img.get("name", "image")
            detected.append(
                {
                    "name": f"ingredient_from_{name}",
                    "form": "unspecified",
                    "estimated_weight_g_or_ml": None,
                    "observed_quantity": None,
                    "spoiling_estimate_days_in_place": None,
                    "confidence": 0.25,
                }
            )
        return json.dumps(
            {
                "detected": detected,
                "summary": f"No image bytes available; returned stub detections. Hints: {hints}",
                "warning": "IMAGE_STORE empty for given handles",
            }
        )

    # This prompt avoids the problem of single type of ingredient group getting outputted only, or too diverse partitions of items.
    system_content = (
        "You are a meticulous kitchen inventory auditor. "
        "You are given one or more photos of a fridge, pantry, or countertop. "
        "Your job is to carefully scan the *entire* image (top to bottom, left to right) "
        "and report the food ingredients and packaged food items that are clearly and decently visible "
        "(e.g. vegetables, meats, fruits, condiments, etc.).\n\n"
        "Important honesty rules:\n"
        "  - Only include items that you can reasonably see in the image.\n"
        "  - Do NOT invent ingredients or categories that are not clearly visible.\n"
        "  - If most items in the scene are from a single broad category (e.g., mostly condiments), "
        "    it is perfectly fine for your output to reflect that.\n"
        "  - If you can clearly see items from multiple categories (e.g., pasta, rice, sauces, canned beans), "
        "    include them all; do not stop after listing just a few.\n\n"
        "DEDUPLICATION AND MERGING (VERY IMPORTANT):\n"
        "  - If multiple items are the same physical ingredient type (e.g., three yellow bell peppers in different spots, "
        "    or two boxes of the same pasta brand), treat them as **one ingredient row** with:\n"
        "      * a single 'name' (e.g., 'yellow bell peppers'),\n"
        "      * a combined 'estimated_weight_g_or_ml' for the TOTAL visible amount,\n"
        "      * an 'observed_quantity' like '3 whole peppers' or '2 boxes', not separate rows.\n"
        "  - Do NOT create separate rows just because the same ingredient appears in different parts of the image.\n"
        "  - Only create separate ingredient rows when the ingredient type is meaningfully different "
        "    (e.g., 'yellow bell peppers' vs 'green bell peppers' vs 'green apples').\n\n"
        "For each distinct ingredient type *after merging duplicates*, produce a JSON object with fields:\n"
        "  - name (concise human-readable name, e.g., 'spaghetti pasta (dried, in box)'),\n"
        "  - form (e.g., 'dried, in box', 'jar', 'bottle', 'loose produce'),\n"
        "  - estimated_weight_g_or_ml (numeric best-guess for the TOTAL visible amount; use your best judgment),\n"
        "  - observed_quantity (e.g., '1 box', '3 jars', '4 whole peppers', 'approx 4 servings'),\n"
        "  - spoiling_estimate_days_in_place (numeric best-guess days before spoilage if left as-is in this state),\n"
        "  - confidence (0–1, your confidence that this row is correctly identified).\n\n"
        "Use best-effort approximations; it is OK to be uncertain, but don't leave fields missing. "
        "Return ONLY a single JSON object with keys:\n"
        "  - 'detected': the list of merged ingredient rows as described above,\n"
        "  - 'summary': a short natural-language summary of the scene and what types of ingredients you found, "
        "    explicitly mentioning how many peppers/apples/etc. you aggregated."
    )

    user_blocks = [
        {
            "type": "text",
            "text": hints
            or "Fridge and pantry overview; detect all visible food items and packages.",
        }
    ] + image_blocks

    try:
        resp = openai_client.chat.completions.create(
            model=vision_model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_blocks},
            ],
        )
        text = resp.choices[0].message.content or ""
        text = text.strip()
        # Try to parse JSON directly
        parsed = json.loads(text)
        # Ensure required keys exist
        detected = parsed.get("detected", [])
        summary = parsed.get("summary", "")
        return json.dumps({"detected": detected, "summary": summary})
    except Exception as e:
        # Fallback: wrap raw text
        return json.dumps(
            {
                "detected": [],
                "summary": f"Vision model call failed: {e}. Raw response: {text if 'text' in locals() else ''}",
                "warning": "vision_detect_ingredients failed to parse JSON",
            }
        )

# ---------------------------------------------------------------------
# Calorie RAG: use RagTool backed by web pages like calories.info
# ---------------------------------------------------------------------

# List of calorie-chart pages bot will use as its knowledge base
CALORIE_SOURCES = [
    "https://www.calories.info/food/meals-dishes",
    "https://www.calories.info/food/vegetables-legumes",
    "https://www.calories.info/food/fruit",
    "https://www.calories.info/food/meat",
    "https://www.calories.info/food/pork",
    "https://www.calories.info/food/mushrooms",
    "https://www.calories.info/food/cheese",
    "https://www.calories.info/food/milk-dairy-products",
    "https://www.calories.info/food/yogurt",
    "https://www.calories.info/food/rice-products",
    "https://www.calories.info/food/cereal",
    "https://www.calories.info/food/bread-rolls-pastries",
    "https://www.calories.info/food/cakes-pies",
    "https://www.calories.info/food/ice-cream",
    "https://www.calories.info/food/sweets-chocolate-cookies-candy",
    "https://www.calories.info/food/fast-food-burgers",
    "https://www.calories.info/food/coffee",
    "https://www.calories.info/food/beer",
    "https://www.calories.info/food/oils-fats",
    "https://www.calories.info/food/herbs-spices-tea",
    "https://www.calories.info/food/salad",
    "https://www.calories.info/food/pizza"
    # Add more here.
]

# Build a RagTool instance and ingest those pages once at startup
calorie_rag_tool = RagTool()
for url in CALORIE_SOURCES:
    calorie_rag_tool.add(data_type="website", url=url)

@tool("calorie_rag_lookup")
def calorie_rag_lookup(query: str) -> str:
    """Call the RagTool but ensuring the input to it is a string."""
    return calorie_rag_tool.run(query=query)


@tool("image_generation_stub")
def image_generation_stub(payload: dict) -> str:
    """Return a placeholder image record for a recipe. Input JSON with recipe_id, name, style_notes."""
    if isinstance(payload, str):
        data = json.loads(payload)
    elif isinstance(payload, dict):
        if "description" in payload and isinstance(payload["description"], str):
            raw = payload["description"]
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"name": raw}
        else:
            data = payload
    else:
        raise ValueError("payload must be a JSON string or dict-like object")

    recipe_id = data.get("recipe_id")
    name = data.get("name") or "recipe"
    style_notes = data.get("style_notes") or "overhead, bright"

    return json.dumps(
        {
            "recipe_id": recipe_id,
            "image_url_or_blob_ref": f"stub://{recipe_id or 'new'}_{name.replace(' ', '_').lower()}.png",
            "style_notes": style_notes,
            "generation_parameters": {"model": "stub"},
        }
    )


@tool("image_generation")
def image_generation(
    recipe_id: str,
    name: str,
    key_ingredients: list[str],
    style_notes: str,
    prompt_override: str | None = "",
    generation_parameters: dict | None = None,
) -> str:
    """Generate a recipe image using gpt-image-1.

    Arguments:
      recipe_id: recipe identifier
      name: recipe name
      key_ingredients: list of key ingredient names
      style_notes: stylistic notes for composition / lighting / mood
      prompt_override: optional full prompt to use instead of auto-building one
      generation_parameters: optional dict with model/size/metadata
    """
    # normalize None into an empty string
    if prompt_override is None:
        prompt_override = ""
    data = {
        "recipe_id": recipe_id,
        "name": name,
        "key_ingredients": key_ingredients,
        "style_notes": style_notes,
        "prompt_override": prompt_override,
        "generation_parameters": generation_parameters or {},
    }

    # Then the rest of bot's logic, using `data` as before:
    recipe_id = data.get("recipe_id")
    name = data.get("name") or "recipe"
    key_ingredients = data.get("key_ingredients") or []
    style_notes = data.get("style_notes") or "overhead, bright light, appetizing plating"
    prompt_override = data.get("prompt_override")

    if prompt_override:
        full_recipe_prompt = prompt_override
    else:
        key_ing = ", ".join(key_ingredients) if key_ingredients else "assorted fresh ingredients"
        full_recipe_prompt = (
            f"High-quality food photograph of {name}. Key ingredients visible: {key_ing}. "
            f"Style: {style_notes}. Realistic lighting, natural textures, appetizing and clean presentation."
        )

    image_model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1-mini")  # TODO: after passing verify, use gpt-image-1
    generation_parameters: Dict[str, Any] = {
        "model": image_model,
        "size": "1024x1024",
    }
    quality = os.getenv("OPENAI_IMAGE_QUALITY")
    if quality:
        generation_parameters["quality"] = quality

    if not openai_client:
        return json.dumps(
            {
                "recipe_id": recipe_id,
                "image_url_or_blob_ref": f"stub://{recipe_id or 'new'}_{name.replace(' ', '_').lower()}.png",
                "style_notes": style_notes,
                "generation_parameters": generation_parameters,
                "warning": "openai client unavailable; returned stub",
            }
        )

    try:
        resp = openai_client.images.generate(
            model=image_model,
            prompt=full_recipe_prompt,
            size=generation_parameters["size"],
            **({"quality": quality} if quality else {}),
        )
        img_data = resp.data[0]

        # Prefer a direct URL from the service if available
        image_url = getattr(img_data, "url", None)

        # Cannot directly return generated image's bytes into context; else overflow!
        if not image_url and hasattr(img_data, "b64_json"):
            # SAFELY handle base64: decode and store in IMAGE_STORE, but DO NOT put base64 into the LLM context.
            img_bytes = base64.b64decode(img_data.b64_json)
            image_url = register_generated_image(img_bytes, recipe_id, name)

    except Exception as e:
        return json.dumps(
            {
                "recipe_id": recipe_id,
                "image_url_or_blob_ref": f"stub://{recipe_id or 'new'}_{name.replace(' ', '_').lower()}.png",
                "style_notes": style_notes,
                "generation_parameters": {**generation_parameters, "error": str(e)},
                "warning": "image generation failed; returned stub",
            }
        )

    return json.dumps(
        {
            "recipe_id": recipe_id,
            "image_url_or_blob_ref": image_url,
            "style_notes": style_notes,
            "generation_parameters": {**generation_parameters, "prompt": full_recipe_prompt},
        }
    )


# ---------------------------------------------------------------------
# DB tools
# ---------------------------------------------------------------------


def build_db_tools(db_path: Path = DEFAULT_DB) -> List[Any]:
    init_db(db_path)

    @tool("db_upsert_ingredients")
    def db_upsert_ingredients(rows_json: str) -> str:
        """Insert or update ingredient rows and return the saved rows as JSON."""
        rows = json.loads(rows_json)
        saved = upsert_ingredients(rows, db_path)
        return json.dumps(saved)

    @tool("db_apply_ingredient_patches")
    def db_apply_ingredient_patches(patches_json: str) -> str:
        """Apply ingredient patch operations (update/delete/insert) and return results."""
        patches = json.loads(patches_json)
        updated = apply_ingredient_patches(patches, db_path)
        return json.dumps(updated)

    @tool("db_upsert_nutrition")
    def db_upsert_nutrition(rows_json: str) -> str:
        """Insert or update nutrition rows and return the saved rows as JSON."""
        rows = json.loads(rows_json)
        saved = upsert_nutrition(rows, db_path)
        return json.dumps(saved)

    @tool("db_apply_nutrition_patches")
    def db_apply_nutrition_patches(patches_json: str) -> str:
        """Apply nutrition patch operations (update/delete/recompute) and return results."""
        patches = json.loads(patches_json)
        updated = apply_nutrition_patches(patches, db_path)
        return json.dumps(updated)

    @tool("db_upsert_recipes")
    def db_upsert_recipes(rows_json: str) -> str:
        """Insert or update recipes (and ingredient usages) and return saved rows."""
        rows = json.loads(rows_json)
        saved = upsert_recipes(rows, db_path)
        return json.dumps(saved)

    @tool("db_apply_recipe_patches")
    def db_apply_recipe_patches(patches_json: str) -> str:
        """Apply recipe patch operations (update/delete/recompute) and return results."""
        patches = json.loads(patches_json)
        updated = apply_recipe_patches(patches, db_path)
        return json.dumps(updated)

    @tool("db_upsert_images")
    def db_upsert_images(rows_json: str) -> str:
        """Insert or update images linked to recipes and return saved rows."""
        rows = json.loads(rows_json)
        saved = upsert_images(rows, db_path)
        return json.dumps(saved)

    @tool("db_fetch_tables")
    def db_fetch_tables(_: str = "") -> str:
        """Fetch all tables (ingredients, nutrition, recipes, recipe_ingredients, images)."""
        return json.dumps(fetch_tables(db_path))

    @tool("session_set_state")
    def session_set_state(payload: str) -> str:
        """Persist a session_state key/value pair."""
        data = json.loads(payload)
        set_state(data.get("key"), data.get("value"), db_path)
        return json.dumps({"key": data.get("key"), "value": data.get("value")})

    @tool("session_get_state")
    def session_get_state(key: str) -> str:
        """Retrieve a session_state value by key."""
        return json.dumps({"key": key, "value": get_state(key, db_path)})

    return [
        db_upsert_ingredients,
        db_apply_ingredient_patches,
        db_upsert_nutrition,
        db_apply_nutrition_patches,
        db_upsert_recipes,
        db_apply_recipe_patches,
        db_upsert_images,
        db_fetch_tables,
        session_set_state,
        session_get_state,
    ]


# ---------------------------------------------------------------------
# KitchenCrew & Agents
# ---------------------------------------------------------------------


class KitchenCrew:
    """Agentic workflow that follows the Agentic Workflow Design document."""

    def __init__(self, db_path: Path = DEFAULT_DB, model: Optional[str] = None):
        self.db_path = db_path
        high_model = model or os.getenv("OPENAI_HIGH_MODEL", "gpt-5.1")
        mid_model = os.getenv("OPENAI_MID_MODEL", "gpt-5-mini")

        # Model sizing: frontier for heavy reasoning/multimodal, mini for structured/utility
        self.ingredients_llm = OpenAICompletion(model=high_model)#, max_completion_tokens=600)
        self.meal_llm = OpenAICompletion(model=high_model)#, max_completion_tokens=800)
        self.orchestrator_llm = OpenAICompletion(model=high_model)#, max_completion_tokens=400)
        self.calorie_llm = OpenAICompletion(model=mid_model)#, max_completion_tokens=400)
        self.image_reason_llm = OpenAICompletion(model=mid_model)#, max_completion_tokens=400)

        self.db_tools = build_db_tools(db_path)
        self.agents = self._build_agents()

    def _build_agents(self) -> Dict[str, Agent]:
        # This version uses the vision detection ingredients tool.
        # ingredients_agent = Agent(
        #     name="Ingredients Extractor",
        #     role="Extract structured ingredients from images and user clarifications.",
        #     goal="Maintain a clean ingredients table with id, name, form, estimated_weight_g_or_ml, observed_quantity, spoiling_estimate_days_in_place, confidence.",
        #     backstory=(
        #         "Meticulous kitchen inventory auditor. Your FIRST step must always be to "
        #         "call the vision_detect_ingredients tool with the provided image handles "
        #         "and hints; do NOT try to hallucinate ingredients without the tool. "
        #         "If the tool doesn't return anything, raise an error. Else, keep trying."
        #         "After the tool returns, clean / refine the rows and persist them via db_upsert_ingredients."
        #     ),
        #     tools=[vision_detect_ingredients_tool, *self.db_tools],
        #     allow_delegation=False,
        #     verbose=True,
        #     llm=self.ingredients_llm,
        # )

        ingredients_agent = Agent(
            name="Ingredients Extractor",
            role="Summarize and maintain the current pantry inventory based on the database.",
            goal=(
                "Read the ingredients table that was already populated by the vision pipeline and "
                "produce clean, structured ingredient rows plus a human-friendly summary."
            ),
            backstory=(
                "You are a meticulous kitchen inventory auditor. The vision extraction step has "
                "ALREADY run before this task and inserted ingredient rows into the database. "
                "You NEVER call the vision_detect_ingredients tool yourself. Instead, you:\n"
                "  - Use db_fetch_tables to read the current 'ingredients' table (and session_get_state "
                "    to read any saved 'ingredients_summary' text),\n"
                "  - Optionally clean or reformat the ingredient rows in memory (e.g., renaming fields),\n"
                "  - Return the ingredient rows and a short natural-language summary to the user.\n"
                "You do not invent new ingredients and you do not write to the database in this task."
            ),
            # Only DB tools are needed here; vision tool is no longer part of this agent's toolbox
            tools=[*self.db_tools],
            allow_delegation=False,
            verbose=True,
            llm=self.ingredients_llm,
        )

        calorie_agent = Agent(
            name="Calorie Provider",
            role="Map each ingredient to standardized nutrition info and keep totals in sync.",
            goal="Produce nutrition rows with calories_per_100g_or_serving, serving size, macro_breakdown, total_calories_for_inventory_amount plus citations.",
            backstory="Nutrition data analyst grounded in citations and spreadsheet-style precision.",
            tools=[calorie_rag_lookup, *self.db_tools],
            allow_delegation=False,
            verbose=True,
            llm=self.calorie_llm,
        )

        meal_agent = Agent(
            name="Meal Planner",
            role="Design candidate recipes that use available inventory and hit calorie targets.",
            goal=(
                "Read the ingredients and nutrition tables and design one or more recipes that:\n"
                "  - use available ingredients (prioritizing those closer to spoiling),\n"
                "  - roughly hit the target calories per person,\n"
                "  - are easy to cook from the current inventory.\n\n"
                "Your job is **only** to plan recipes and return them as structured JSON. "
                "You DO NOT call any database tools to write recipes; another part of the "
                "system will persist them."
            ),
            backstory=(
                "Pragmatic home-chef who scales portions, reduces waste, and tweaks without "
                "rebuilding everything. You read from the existing ingredients and nutrition "
                "tables, then propose structured recipes as JSON."
            ),
            # Only need read access (db_fetch_tables) for planning. If want,
            # can keep all db tools here; the important part is the instructions above.
            tools=self.db_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.meal_llm,
        )

        image_agent = Agent(
            name="Image Generator",
            role="Generate plausible images for each recipe and refresh them on feedback.",
            goal="Store images tied to recipe_id with style_notes and generation_parameters for reproducibility.",
            backstory="Food stylist powered by an image model; keeps track of which image belongs to which recipe.",
            tools=[image_generation, *self.db_tools],
            allow_delegation=False,
            verbose=True,
            llm=self.image_reason_llm,
        )

        orchestrator = Agent(
            name="Orchestrator",
            role="Coordinate ingredients -> calories -> meals -> images while routing feedback.",
            goal="Recompute only what is needed, track session state, and assemble recipe cards with matching images.",
            backstory="Head chef and floor manager who assigns the right sous-chef and keeps plates aligned with photos and calorie counts.",
            tools=self.db_tools,
            allow_delegation=False,
            verbose=True,
            llm=self.orchestrator_llm,
        )

        return {
            "ingredients": ingredients_agent,
            "calories": calorie_agent,
            "meals": meal_agent,
            "images": image_agent,
            "orchestrator": orchestrator,
        }

    def _build_extract_task(self) -> Task:
        # This version uses the vision detection ingredients tool.
        # return Task(
        #     name="ExtractIngredientsFromImages",
        #     description=(
        #         "You are given uploaded fridge/pantry images. "
        #         "The input includes image handles: {{images}} and user hints: '{{hints}}'. "
        #         "Your goal is to honestly and thoroughly extract the food ingredients and packaged food items "
        #         "that are actually visible in those images.\n\n"
        #         "1) Construct a JSON payload with keys 'images' and 'hints', where 'images' "
        #         "   is exactly the list of handle objects from {{images}} (do NOT stringify it), "
        #         "   and 'hints' is the string from {{hints}}.\n"
        #         "2) Call the vision_detect_ingredients tool ONCE with that JSON payload to detect ingredients. "
        #         "   Do NOT invent items that are not decently visible; if the scene is mostly one type of ingredient "
        #         "   (e.g., a shelf of pasta boxes), it is fine to return mostly that type.\n"
        #         "3) Optionally clean/refine the returned ingredient rows (e.g., merge obvious duplicates, tweak names) "
        #         "   without adding things that aren't in the image.\n"
        #         "4) Persist the final ingredient rows via db_upsert_ingredients.\n\n"
        #         "Return a JSON payload with keys 'ingredients' (the final rows you persisted) and "
        #         "'summary' (short sentences for the user describing what you found)."
        #     ),
        #     expected_output="JSON payload with ingredients and summary.",
        #     agent=self.agents["ingredients"],
        # )

        return Task(
            name="ExtractIngredientsFromImages",
            description=(
                "You are given uploaded fridge/pantry images, but the vision extraction step "
                "has already run BEFORE this task and stored ingredient rows in the database.\n\n"
                "Your job now is to summarize what was found:\n"
                "1) Use db_fetch_tables to read the current 'ingredients' table.\n"
                "   - Do NOT call the vision_detect_ingredients tool; the ingredients have already been "
                "     detected and inserted.\n"
                "2) Optionally use session_get_state with key 'ingredients_summary' to retrieve any "
                "   previously saved summary text from the Phase 0 vision step.\n"
                "3) Optionally make small in-memory cleanups to the ingredient rows (e.g., renaming fields "
                "   or merging obvious duplicates), but do NOT invent new ingredients and do NOT write "
                "   anything back to the database in this task.\n"
                "4) Return a JSON payload with keys:\n"
                "   - 'ingredients': the list of ingredient rows as you want to show them to the user,\n"
                "   - 'summary': a short natural-language description of what is currently in the inventory.\n"
                "Use the existing database contents as the source of truth."
            ),
            expected_output="JSON payload with ingredients and summary.",
            agent=self.agents["ingredients"],
        )

    def _build_downstream_tasks(self, extract: Task) -> List[Task]:
        calories = Task(
            name="FetchCaloriesForIngredients",
            description=(
                "Using the confirmed ingredients from the previous task, map each ingredient to nutrition facts.\n\n"
                "1. Use db_fetch_tables to read the current 'ingredients' table. For each row, note its ingredient_id, name, form, "
                "   and estimated_weight_g_or_ml (this is the inventory amount in grams or ml).\n"
                "2. Build a SINGLE composite query string that lists ALL ingredients and their ids, like:\n"
                "   \"You are a nutrition database. Here are my ingredients with ids and inventory amounts:\n"
                "    1: rolled oats (dry), 200 g in inventory\n"
                "    2: whole milk (2%), 500 g in inventory\n"
                "    3: banana, 150 g in inventory\n"
                "    ...\n"
                "    For EACH ingredient, return an object with fields:\n"
                "      - ingredient_id (matching the id above),\n"
                "      - calories_per_100g_or_serving (float),\n"
                "      - standard_serving_size (string),\n"
                "      - macro_breakdown: {\"protein_g\": ..., \"fat_g\": ..., \"carbs_g\": ...} if available,\n"
                "      - source: short text describing which tables/websites you used.\n"
                "    Return ONLY a JSON array of these objects, no extra commentary.\"\n"
                "3. Call the RAG calorie knowledge tool *exactly once* using Action Input of the form:\n"
                "   {\"query\": \"<your composite query string>\"}.\n"
                "   Do NOT wrap the query in a nested object like {\"description\": ..., \"type\": ...}; the 'query' value must be a plain string.\n"
                "4. Parse the JSON array returned by calorie_rag_lookup. For each object, compute\n"
                "   total_calories_for_inventory_amount = (calories_per_100g_or_serving * inventory_amount_in_grams / 100),\n"
                "   where inventory_amount_in_grams comes from the ingredients table.\n"
                "5. For each ingredient, construct a nutrition row with fields:\n"
                "   ingredient_id, calories_per_100g_or_serving, standard_serving_size, macro_breakdown (if available), "
                "   total_calories_for_inventory_amount, source, and updated_at (use a reasonable timestamp string).\n"
                "6. Persist the full list of nutrition rows in one call via db_upsert_nutrition, passing rows_json as a JSON array of these rows.\n"
                "If the knowledge tool does not return usable information for a particular ingredient, estimate its values based on similar foods "
                "and mark the source as 'estimated'."

                # This query asks for calorie ingredient by ingredient; too slow!
                # "Using the confirmed ingredients from the previous task, map each to nutrition facts. "
                # "You have access to a RAG calorie knowledge tool (calorie_rag_lookup) that has indexed several calorie-chart websites "
                # "(including calories.info). For each ingredient, query the tool with a natural-language question like: "
                # "\"According to your sources, how many calories per 100 g does [INGREDIENT_NAME] have? Also give typical serving size if available.\" "
                # "From the tool's answer, produce nutrition rows with ingredient_id, calories_per_100g_or_serving, standard_serving_size, "
                # "macro_breakdown if possible, total_calories_for_inventory_amount, and a 'source' string describing where the value came from. "
                # "If the tool does not return useful info for an ingredient, use your knowledge to make a best-effort estimate, and note 'estimated' in the source. "
                # "Persist the rows via db_upsert_nutrition."
            ),
            expected_output="JSON list of nutrition rows with citations.",
            agent=self.agents["calories"],
            context=[extract],
        )

        plan = Task(
            name="PlanMeals",
            description=(
                "Design one candidate recipe that uses available ingredients and hits the calorie targets.\n"
                "User inputs: meal_time={{meal_time}}, target_calories_per_person={{target_calories_per_person}}, "
                "num_people={{num_people}}, constraints={{constraints}}.\n\n"
                "1) Use db_fetch_tables to read the current 'ingredients' and 'nutrition' tables.\n"
                "2) Choose ingredients that are available and, when possible, closer to spoiling "
                "(shorter spoiling_estimate_days_in_place), and design ONE recipe for the given "
                "meal_time that aims for target_calories_per_person * num_people total calories.\n"
                "3) Represent the recipe as a JSON object with fields:\n"
                "   - recipe_id (a new unique id, e.g. 1001 or 'new_breakfast_001'),\n"
                "   - name,\n"
                "   - meal_time,\n"
                "   - servings (num_people),\n"
                "   - per_person_calories (approximate),\n"
                "   - total_calories (approximate),\n"
                "   - ingredient_usages: a list of {\"ingredient_id\": <int>, "
                "     \"amount_g_or_ml_per_serving\": <float>} referencing existing ingredient_ids,\n"
                "   - instructions: ordered list of steps,\n"
                "   - dietary_tags: list of strings.\n"
                "4) Build a SINGLE JSON array called 'recipes' containing this one recipe object, "
                "and return a JSON object of the form:\n"
                "   {\"recipes\": [<your recipe object>], "
                "    \"summary\": <short explanation of calorie fit and spoilage strategy>}.\n\n"
                "You DO NOT call any database tools to write recipes in this task; you only read "
                "tables and return JSON."
            ),
            expected_output=(
                "A JSON object with keys 'recipes' (a list with one recipe object) and "
                "'summary' (a short natural-language explanation)."
            ),
            agent=self.agents["meals"],
            context=[extract, calories],
        )

        images = Task(
            name="GenerateRecipeImages",
            description=(
                "For each recipe from PlanMeals, generate an image record with recipe_id, image_url_or_blob_ref, style_notes, generation_parameters. "
                "Use image_generation (or image_generation_stub if configured). Persist via db_upsert_images."
            ),
            expected_output="JSON list of image records linked to recipe_id.",
            agent=self.agents["images"],
            context=[plan],
        )

        compose = Task(
            name="ComposeRecipesAndImagesForDisplay",
            description=(
                "Assemble UI-ready recipe cards by combining recipes and their best images. "
                "Each card must include: recipe_id, name, meal_time, servings, "
                "per_person_calories, total_calories, key_ingredients, uses_soon_to_spoil, "
                "image_url_or_blob_ref, short_description.\n\n"
                "Return **only** a single valid JSON object with this exact top-level structure:\n"
                "{\n"
                '  \"recipe_cards\": [\n'
                "    {\n"
                '      \"recipe_id\": int,\n'
                '      \"name\": str,\n'
                '      \"meal_time\": str,\n'
                '      \"servings\": int,\n'
                '      \"per_person_calories\": float,\n'
                '      \"total_calories\": float,\n'
                '      \"key_ingredients\": [str, ...],\n'
                '      \"uses_soon_to_spoil\": [str, ...],\n'
                '      \"image_url_or_blob_ref\": str,\n'
                '      \"short_description\": str\n'
                "    }\n"
                "  ],\n"
                '  \"summary\": str  // brief natural-language comparison of the recipe cards\n'
                "}\n\n"
                "Do NOT wrap the JSON in markdown or add any text outside the JSON object."
            ),
            expected_output=(
                "A single JSON object: {\"recipe_cards\": [...], \"summary\": \"...\"} "
                "with the fields described in the task description, and nothing else."
            ),
            agent=self.agents["orchestrator"],
            context=[plan, images],
        )


        return [calories, plan, images, compose]

    def run_primary_flow(
        self,
        images: List[Dict[str, Any]],
        hints: str,
        meal_time: str,
        target_calories_per_person: float,
        num_people: int,
        constraints: str = "",
    ) -> Dict[str, Any]:
    
        # Need to manually run vision tool! Else the agent is too lazy to run it via prompting.
        # Phase 0: vision call and ingredient upsert (outside CrewAI)
        if images:
            vision_payload = {"images": images, "hints": hints}
            vision_raw = vision_detect_ingredients(vision_payload)
            try:
                vision_obj = json.loads(vision_raw)
            except Exception:
                vision_obj = {}
            detected = (
                vision_obj.get("detected")
                or vision_obj.get("ingredients")
                or []
            )
            summary = vision_obj.get("summary", "")

            # Write directly into the DB
            upsert_ingredients(detected, self.db_path)
            set_state("ingredients_summary", summary, self.db_path)

        # Phase 1: extract ingredients (only task that sees images)
        extract_task = self._build_extract_task()
        extract_crew = Crew(
            agents=[self.agents["ingredients"]],
            tasks=[extract_task],
            process=Process.sequential,
            verbose=True,
        )
        extract_result = extract_crew.kickoff(inputs={"images": images, "hints": hints})

        # Phase 2: downstream tasks without images in the input
        downstream_tasks = self._build_downstream_tasks(extract_task)
        downstream_crew = Crew(
            agents=[
                self.agents["calories"],
                self.agents["meals"],
                self.agents["images"],
                self.agents["orchestrator"],
            ],
            tasks=downstream_tasks,
            process=Process.sequential,
            verbose=True,
        )
        downstream_inputs = {
            "meal_time": meal_time,
            "target_calories_per_person": target_calories_per_person,
            "num_people": num_people,
            "constraints": constraints,
        }
        downstream_result = downstream_crew.kickoff(inputs=downstream_inputs)

        # Find the PlanMeals task object
        plan_task = next(
            (t for t in downstream_tasks if t.name == "PlanMeals"),
            None,
        )
        if plan_task is not None:
            raw_plan = getattr(plan_task, "output", None)

            # CrewAI Task.output is often a TaskOutput with a `.raw` field
            plan_payload = getattr(raw_plan, "raw", raw_plan)

            try:
                if isinstance(plan_payload, str):
                    plan_obj = json.loads(plan_payload)
                else:
                    plan_obj = plan_payload
            except Exception:
                plan_obj = None

            # Two possible formats we’ve seen:
            # 1) {"rows_json": "<JSON array string>"}
            # 2) {"recipes": [ {...}, ... ], "summary": "..."}
            if isinstance(plan_obj, dict):
                recipe_rows: List[Dict[str, Any]] = []

                if "rows_json" in plan_obj:
                    try:
                        recipe_rows = json.loads(plan_obj["rows_json"])
                    except Exception:
                        recipe_rows = []

                elif "recipes" in plan_obj and isinstance(plan_obj["recipes"], list):
                    recipe_rows = plan_obj["recipes"]

                if recipe_rows:
                    # Persist recipes + recipe_ingredients via DB helper
                    upsert_recipes(recipe_rows, self.db_path)

        task_outputs = {"ExtractIngredientsFromImages": getattr(extract_task, "output", None)}
        for task in downstream_tasks:
            task_outputs[task.name] = getattr(task, "output", None)
        return {"result": {"extract": extract_result, "downstream": downstream_result}, "task_outputs": task_outputs}

    def run_ingredient_feedback(self, feedback: str) -> Any:
        task = Task(
            name="UpdateIngredientsFromUserFeedback",
            description=(
                "User feedback: {{feedback}}. Use current ingredients table from db_fetch_tables to apply updates or deletes. "
                "Return patch operations applied and persist via db_apply_ingredient_patches."
            ),
            expected_output="JSON list of patch operations executed.",
            agent=self.agents["ingredients"],
        )
        crew = Crew(
            agents=[self.agents["ingredients"]],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff(inputs={"feedback": feedback})

    def run_nutrition_feedback(self, feedback: str) -> Any:
        task = Task(
            name="UpdateCaloriesFromUserFeedback",
            description=(
                "User feedback: {{feedback}}. Adjust nutrition rows accordingly using db_apply_nutrition_patches and recompute totals."
            ),
            expected_output="JSON list of nutrition patch operations executed.",
            agent=self.agents["calories"],
        )
        crew = Crew(
            agents=[self.agents["calories"]],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff(inputs={"feedback": feedback})

    def run_recipe_feedback(self, feedback: str) -> Any:
        task = Task(
            name="RefineRecipeFromUserFeedback",
            description=(
                "User feedback: {{feedback}}. Modify the targeted recipe using db_apply_recipe_patches (update_ingredient_usage, recompute_calories, etc.)."
            ),
            expected_output="Patch operations and updated recipe fields.",
            agent=self.agents["meals"],
        )
        crew = Crew(
            agents=[self.agents["meals"]],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff(inputs={"feedback": feedback})

    def regenerate_image(self, feedback: str) -> Any:
        task = Task(
            name="RegenerateOrVaryImage",
            description=(
                "User feedback: {{feedback}}. Use image_generation (or image_generation_stub) and db_upsert_images to attach a new image to the referenced recipe_id."
            ),
            expected_output="New image record JSON tied to recipe_id.",
            agent=self.agents["images"],
        )
        crew = Crew(
            agents=[self.agents["images"]],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        return crew.kickoff(inputs={"feedback": feedback})

    def fetch_tables(self) -> Dict[str, Any]:
        return fetch_tables(self.db_path)


# ---------------------------------------------------------------------
# Image encoding helper used by the Streamlit UI
# ---------------------------------------------------------------------


def encode_uploaded_images(files: List[bytes], names: List[str]) -> List[Dict[str, Any]]:
    """
    Store raw bytes in IMAGE_STORE and return small image handles to the LLM.

    Output is a list of:
      { "image_id": "img_0", "name": "fridge.jpg" }
    """
    payload: List[Dict[str, Any]] = []
    IMAGE_STORE.clear()
    for idx, (name, content) in enumerate(zip(names, files)):
        image_id = f"img_{idx}"
        IMAGE_STORE[image_id] = content
        payload.append({"image_id": image_id, "name": name})
    return payload


if __name__ == "__main__":
    crew = KitchenCrew()
    print("Agents ready:", list(crew.agents))

