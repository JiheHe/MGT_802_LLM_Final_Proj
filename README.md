# Agentic Kitchen Planner

A small **calorie-aware fridge clean-out assistant**.

You upload one or more photos of your fridge / pantry, specify a meal time and target calories per person, and the app:

1. Uses a vision model to infer what ingredients you have.
2. Calls a RAG-based calorie lookup to estimate kcal + macros for each ingredient.
3. Plans a breakfast/lunch/dinner recipe that tries to:
   - Hit your target calories per person.
   - Work with your specified constraints.
   - Prioritize ingredients that are closer to spoiling.
4. Generates a plausible dish image and shows everything as a **recipe card**.

All state (ingredients, nutrition, recipes, images) is stored in a local SQLite DB (`db.sqlite`) and shown in a **Current tables** section at the bottom of the app.

---

## 1. Prerequisites

- Python **3.10+** (3.11 is fine)
- An **OpenAI API key** with access to:
  - `gpt-5.1` (or whatever you configure as `OPENAI_HIGH_MODEL`)
  - `gpt-5-mini` (default `OPENAI_MID_MODEL`)
  - `gpt-5.1` or `gpt-4o` with vision for ingredient extraction
  - `gpt-image-1` or `gpt-image-1-mini` for image generation (optional; the app falls back to stub images if image models aren't available)

> If you don't have image-model access yet, the app will still run; it will just store stub image metadata and you'll see a placeholder instead of a real generated image.

---

## 2. Setup

1. **Clone or download** this repo:

   ```bash
   git clone <YOUR_REPO_URL> MGT_802_LLM_FINAL_PROJ
   cd MGT_802_LLM_FINAL_PROJ
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv llm_env
   # macOS / Linux
   source llm_env/bin/activate
   # Windows (PowerShell)
   # .\llm_env\Scripts\Activate.ps1
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key for the current shell:**

   ```bash
   # macOS / Linux
   export OPENAI_API_KEY="sk-..."
   # Windows (PowerShell)
   # $env:OPENAI_API_KEY="sk-..."
   ```

5. **(Optional) Override default model names via env vars:**

   ```bash
   export OPENAI_HIGH_MODEL="gpt-5.1"
   export OPENAI_MID_MODEL="gpt-5-mini"
   export OPENAI_VISION_MODEL="gpt-5.1"
   export OPENAI_IMAGE_MODEL="gpt-image-1-mini"
   ```

---

## 3. Running the app

From the repo root:

```bash
streamlit run app.py
```

You should see something like:

```
Local URL:   http://localhost:8501
Network URL: http://<your-ip>:8501
```

Open the Local URL in your browser.

Note: `app.py` currently resets the DB on startup by deleting any existing `db.sqlite` and calling `init_db(DEFAULT_DB)`, so each run starts with a fresh, empty pantry.

---

## 4. Using the UI

Once the app is running:

- Configure the run in the left sidebar:
  - Optional hints - e.g. ignore canned drinks; focus on proteins and veggies. These hints are passed into the vision extractor to bias what it pays attention to.
  - Meal time - breakfast, lunch, or dinner.
  - Target calories per person - e.g. 650.
  - Number of people - how many servings to plan for.
  - Constraints - free-text such as high protein, no dairy, must use spinach, etc.
- Upload fridge / pantry photos:
  - Click "Upload fridge/pantry/counter images".
  - Select one or more `.jpg` / `.jpeg` / `.png` files (full-fridge or pantry shots are ideal).
- Run the full pipeline:
  - Click "Run full workflow".
  - The app will:
    - Call the vision tool once to detect ingredients and write them into the `ingredients` table.
    - Build a single composite query and call the calorie RAG once to populate the `nutrition` table.
    - Plan one recipe for the chosen meal/time/target calories and insert it into `recipes` + `recipe_ingredients`.
    - Call the image generator and store a `gen://...` handle plus metadata in the `images` table.
    - Assemble a `recipe_cards` JSON object that pairs each recipe with its best image.
- Inspect the results:
  - After the run finishes you'll see:
    - Task outputs - raw JSON from each CrewAI task, useful for debugging.
    - Crew result - the final `{"recipe_cards": [...], "summary": ...}` object the orchestrator produced.
    - Recipe cards - a human-friendly section that shows, for each card:
      - Recipe name
      - Short description
      - Generated image, loaded as follows:
        - `stub://...` -> a static placeholder image (if you keep `assets/placeholder_bowl.png`).
        - `gen://...` -> looks up the corresponding bytes in the in-memory `IMAGE_STORE` and displays them.
        - Any other string -> treated as a direct URL or file path and passed to `st.image`.
- Current tables:
  - At the bottom, the app shows the contents of the backing SQLite DB:
    - `ingredients` - what the vision step inferred and deduplicated from your photos.
    - `nutrition` - kcal per 100 g, macros, and total kcal for the inventory amount, sourced via calories.info + USDA-like assumptions.
    - `recipes` - the planned meal with servings, per-person calories, total calories, and instructions.
    - `recipe_ingredients` - the gram/ml usage per ingredient per serving.
    - `images` - mapping from `recipe_id` to `image_url_or_blob_ref` (stub, gen handle, or URL) plus `generation_parameters`.

---

## 5. Typical end-to-end flow

1. Start the app (`streamlit run app.py`).
2. Upload one or more fridge / pantry photos.
3. Choose e.g. breakfast, 650 kcal per person, 2 people, and optional constraints.
4. Click **Run full workflow**.
5. Scroll down to:
   - Verify the `ingredients` table (e.g. yellow/red/green bell peppers, green apples).
   - Inspect the `nutrition` table (calories + macros per 100 g and per-inventory).
   - See the planned recipe (e.g. Sweet Pepper & Green Apple Breakfast Hash), with precise grams per serving.
   - View the generated image illustrating that recipe.
