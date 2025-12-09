from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "db.sqlite"


@contextmanager
def get_conn(db_path: Path = DEFAULT_DB):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DEFAULT_DB) -> None:
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ingredients (
                ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                form TEXT,
                estimated_weight_g_or_ml REAL,
                observed_quantity REAL,
                spoiling_estimate_days_in_place REAL,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS nutrition (
                ingredient_id INTEGER PRIMARY KEY,
                calories_per_100g_or_serving REAL,
                standard_serving_size TEXT,
                macro_protein_g REAL,
                macro_fat_g REAL,
                macro_carbs_g REAL,
                total_calories_for_inventory_amount REAL,
                source TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(ingredient_id) REFERENCES ingredients(ingredient_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recipes (
                recipe_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                meal_time TEXT,
                servings INTEGER,
                per_person_calories REAL,
                total_calories REAL,
                instructions TEXT,
                dietary_tags TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recipe_ingredients (
                recipe_id INTEGER,
                ingredient_id INTEGER,
                amount_g_or_ml_per_serving REAL,
                PRIMARY KEY (recipe_id, ingredient_id),
                FOREIGN KEY(recipe_id) REFERENCES recipes(recipe_id),
                FOREIGN KEY(ingredient_id) REFERENCES ingredients(ingredient_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                recipe_id INTEGER,
                image_url_or_blob_ref TEXT,
                style_notes TEXT,
                generation_parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(recipe_id) REFERENCES recipes(recipe_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS session_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )


def rows_to_dicts(cursor: sqlite3.Cursor) -> List[Dict[str, Any]]:
    return [dict(r) for r in cursor.fetchall()]


def upsert_ingredients(rows: Iterable[Dict[str, Any]], db_path: Path = DEFAULT_DB) -> List[Dict[str, Any]]:
    saved: List[Dict[str, Any]] = []
    allowed_fields = {
        "name",
        "form",
        "estimated_weight_g_or_ml",
        "observed_quantity",
        "spoiling_estimate_days_in_place",
        "confidence",
    }
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        for row in rows:
            ingredient_id = row.get("ingredient_id")
            payload = {
                k: row.get(k) for k in allowed_fields
            }
            if ingredient_id:
                cur.execute(
                    """
                    UPDATE ingredients
                    SET name=:name, form=:form, estimated_weight_g_or_ml=:estimated_weight_g_or_ml,
                        observed_quantity=:observed_quantity, spoiling_estimate_days_in_place=:spoiling_estimate_days_in_place,
                        confidence=:confidence
                    WHERE ingredient_id=:ingredient_id
                    """,
                    {**payload, "ingredient_id": ingredient_id},
                )
            else:
                cur.execute(
                    """
                    INSERT INTO ingredients (name, form, estimated_weight_g_or_ml, observed_quantity, spoiling_estimate_days_in_place, confidence)
                    VALUES (:name, :form, :estimated_weight_g_or_ml, :observed_quantity, :spoiling_estimate_days_in_place, :confidence)
                    """,
                    payload,
                )
                ingredient_id = cur.lastrowid
            cur.execute("SELECT * FROM ingredients WHERE ingredient_id=?", (ingredient_id,))
            row_out = dict(cur.fetchone())
            saved.append(row_out)
    return saved


def apply_ingredient_patches(patches: Iterable[Dict[str, Any]], db_path: Path = DEFAULT_DB) -> List[Dict[str, Any]]:
    allowed_fields = {
        "name",
        "form",
        "estimated_weight_g_or_ml",
        "observed_quantity",
        "spoiling_estimate_days_in_place",
        "confidence",
    }
    results: List[Dict[str, Any]] = []
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        for patch in patches:
            action = patch.get("action")
            ingredient_id = patch.get("ingredient_id")
            if action == "delete" and ingredient_id:
                cur.execute("DELETE FROM ingredients WHERE ingredient_id=?", (ingredient_id,))
                cur.execute("DELETE FROM nutrition WHERE ingredient_id=?", (ingredient_id,))
                cur.execute("DELETE FROM recipe_ingredients WHERE ingredient_id=?", (ingredient_id,))
                results.append({"ingredient_id": ingredient_id, "status": "deleted"})
            elif action == "update" and ingredient_id and patch.get("field") in allowed_fields:
                field = patch["field"]
                cur.execute(
                    f"UPDATE ingredients SET {field}=? WHERE ingredient_id=?",
                    (patch.get("new_value"), ingredient_id),
                )
                results.append({"ingredient_id": ingredient_id, "field": field, "status": "updated"})
            elif action == "insert":
                inserted = upsert_ingredients([patch], db_path)
                results.extend(inserted)
    return results


def upsert_nutrition(rows: Iterable[Dict[str, Any]], db_path: Path = DEFAULT_DB) -> List[Dict[str, Any]]:
    saved: List[Dict[str, Any]] = []
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        for row in rows:
            ingredient_id = row.get("ingredient_id")
            if ingredient_id is None:
                continue
            payload = {
                "ingredient_id": ingredient_id,
                "calories_per_100g_or_serving": row.get("calories_per_100g_or_serving"),
                "standard_serving_size": row.get("standard_serving_size"),
                "macro_protein_g": row.get("macro_breakdown", {}).get("protein_g") if isinstance(row.get("macro_breakdown"), dict) else row.get("macro_protein_g"),
                "macro_fat_g": row.get("macro_breakdown", {}).get("fat_g") if isinstance(row.get("macro_breakdown"), dict) else row.get("macro_fat_g"),
                "macro_carbs_g": row.get("macro_breakdown", {}).get("carbs_g") if isinstance(row.get("macro_breakdown"), dict) else row.get("macro_carbs_g"),
                "total_calories_for_inventory_amount": row.get("total_calories_for_inventory_amount"),
                "source": row.get("source") or row.get("citation"),
            }
            cur.execute(
                """
                INSERT INTO nutrition (
                    ingredient_id, calories_per_100g_or_serving, standard_serving_size, macro_protein_g,
                    macro_fat_g, macro_carbs_g, total_calories_for_inventory_amount, source
                ) VALUES (:ingredient_id, :calories_per_100g_or_serving, :standard_serving_size, :macro_protein_g,
                    :macro_fat_g, :macro_carbs_g, :total_calories_for_inventory_amount, :source)
                ON CONFLICT(ingredient_id) DO UPDATE SET
                    calories_per_100g_or_serving=excluded.calories_per_100g_or_serving,
                    standard_serving_size=excluded.standard_serving_size,
                    macro_protein_g=excluded.macro_protein_g,
                    macro_fat_g=excluded.macro_fat_g,
                    macro_carbs_g=excluded.macro_carbs_g,
                    total_calories_for_inventory_amount=excluded.total_calories_for_inventory_amount,
                    source=excluded.source,
                    updated_at=CURRENT_TIMESTAMP
                """,
                payload,
            )
            cur.execute("SELECT * FROM nutrition WHERE ingredient_id=?", (ingredient_id,))
            saved.append(dict(cur.fetchone()))
    return saved


def apply_nutrition_patches(patches: Iterable[Dict[str, Any]], db_path: Path = DEFAULT_DB) -> List[Dict[str, Any]]:
    allowed_fields = {
        "calories_per_100g_or_serving",
        "standard_serving_size",
        "macro_protein_g",
        "macro_fat_g",
        "macro_carbs_g",
        "total_calories_for_inventory_amount",
        "source",
    }
    results: List[Dict[str, Any]] = []
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        for patch in patches:
            action = patch.get("action")
            ingredient_id = patch.get("ingredient_id")
            if action == "delete" and ingredient_id:
                cur.execute("DELETE FROM nutrition WHERE ingredient_id=?", (ingredient_id,))
                results.append({"ingredient_id": ingredient_id, "status": "deleted"})
            elif action == "update" and ingredient_id and patch.get("field") in allowed_fields:
                field = patch["field"]
                cur.execute(
                    f"UPDATE nutrition SET {field}=? WHERE ingredient_id=?",
                    (patch.get("new_value"), ingredient_id),
                )
                results.append({"ingredient_id": ingredient_id, "field": field, "status": "updated"})
            elif action == "recompute_total" and ingredient_id:
                recompute_calories_for_ingredient(ingredient_id, conn)
                results.append({"ingredient_id": ingredient_id, "status": "recomputed"})
    return results

def upsert_recipes(recipes: Iterable[Dict[str, Any]], db_path: Path = DEFAULT_DB) -> List[Dict[str, Any]]:
    saved: List[Dict[str, Any]] = []
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        for recipe in recipes:
            # Normalize recipe_id: only treat real ints as "existing"
            raw_id = recipe.get("recipe_id")
            recipe_id: Optional[int]
            if isinstance(raw_id, int):
                recipe_id = raw_id
            else:
                # Strings like "new_breakfast_001" or "1001" -> treat as new recipe
                recipe_id = None

            payload = {
                "name": recipe.get("name"),
                "meal_time": recipe.get("meal_time"),
                "servings": recipe.get("servings"),
                "per_person_calories": recipe.get("per_person_calories"),
                "total_calories": recipe.get("total_calories"),
                "instructions": json.dumps(recipe.get("instructions", []))
                if isinstance(recipe.get("instructions"), list)
                else recipe.get("instructions"),
                "dietary_tags": ",".join(recipe.get("dietary_tags", []))
                if isinstance(recipe.get("dietary_tags"), list)
                else recipe.get("dietary_tags"),
            }

            if recipe_id is not None:
                cur.execute(
                    """
                    UPDATE recipes SET
                        name=:name,
                        meal_time=:meal_time,
                        servings=:servings,
                        per_person_calories=:per_person_calories,
                        total_calories=:total_calories,
                        instructions=:instructions,
                        dietary_tags=:dietary_tags
                    WHERE recipe_id=:recipe_id
                    """,
                    {**payload, "recipe_id": recipe_id},
                )
                # If no row was updated (e.g. bogus id), fall back to INSERT
                if cur.rowcount == 0:
                    cur.execute(
                        """
                        INSERT INTO recipes (name, meal_time, servings, per_person_calories,
                                             total_calories, instructions, dietary_tags)
                        VALUES (:name, :meal_time, :servings, :per_person_calories,
                                :total_calories, :instructions, :dietary_tags)
                        """,
                        payload,
                    )
                    recipe_id = cur.lastrowid
            else:
                cur.execute(
                    """
                    INSERT INTO recipes (name, meal_time, servings, per_person_calories,
                                         total_calories, instructions, dietary_tags)
                    VALUES (:name, :meal_time, :servings, :per_person_calories,
                            :total_calories, :instructions, :dietary_tags)
                    """,
                    payload,
                )
                recipe_id = cur.lastrowid

            usages = recipe.get("ingredient_usages") or []
            cur.execute("DELETE FROM recipe_ingredients WHERE recipe_id=?", (recipe_id,))
            for usage in usages:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO recipe_ingredients (recipe_id, ingredient_id, amount_g_or_ml_per_serving)
                    VALUES (?, ?, ?)
                    """,
                    (
                        recipe_id,
                        usage.get("ingredient_id"),
                        usage.get("amount_g_or_ml_per_serving"),
                    ),
                )

            cur.execute("SELECT * FROM recipes WHERE recipe_id=?", (recipe_id,))
            row = cur.fetchone()
            if row is None:
                # Safety guard; shouldn't happen now, but avoid crashing
                continue

            row_out = dict(row)
            row_out["ingredient_usages"] = usages
            saved.append(row_out)
            recompute_recipe_calories(recipe_id, conn)

    return saved



def apply_recipe_patches(patches: Iterable[Dict[str, Any]], db_path: Path = DEFAULT_DB) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        for patch in patches:
            action = patch.get("action")
            recipe_id = patch.get("recipe_id")
            if not recipe_id:
                continue
            if action == "delete":
                cur.execute("DELETE FROM recipes WHERE recipe_id=?", (recipe_id,))
                cur.execute("DELETE FROM recipe_ingredients WHERE recipe_id=?", (recipe_id,))
                cur.execute("DELETE FROM images WHERE recipe_id=?", (recipe_id,))
                results.append({"recipe_id": recipe_id, "status": "deleted"})
            elif action == "update" and patch.get("field"):
                field = patch["field"]
                cur.execute(
                    f"UPDATE recipes SET {field}=? WHERE recipe_id=?",
                    (patch.get("new_value"), recipe_id),
                )
                results.append({"recipe_id": recipe_id, "field": field, "status": "updated"})
            elif action == "update_ingredient_usage" and patch.get("ingredient_id"):
                cur.execute(
                    "INSERT OR REPLACE INTO recipe_ingredients (recipe_id, ingredient_id, amount_g_or_ml_per_serving) VALUES (?, ?, ?)",
                    (
                        recipe_id,
                        patch.get("ingredient_id"),
                        patch.get("new_amount_g_or_ml_per_serving"),
                    ),
                )
                recompute_recipe_calories(recipe_id, conn)
                results.append({"recipe_id": recipe_id, "ingredient_id": patch.get("ingredient_id"), "status": "updated"})
            elif action == "recompute_calories":
                recompute_recipe_calories(recipe_id, conn)
                results.append({"recipe_id": recipe_id, "status": "recomputed"})
    return results


def recompute_calories_for_ingredient(ingredient_id: int, conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
        UPDATE nutrition
        SET total_calories_for_inventory_amount = CASE
            WHEN calories_per_100g_or_serving IS NOT NULL AND i.estimated_weight_g_or_ml IS NOT NULL THEN
                calories_per_100g_or_serving * (i.estimated_weight_g_or_ml / 100.0)
            ELSE total_calories_for_inventory_amount
        END,
        updated_at=CURRENT_TIMESTAMP
        FROM ingredients i
        WHERE nutrition.ingredient_id=i.ingredient_id AND nutrition.ingredient_id=?
    """, (ingredient_id,))


def recompute_recipe_calories(recipe_id: int, conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ri.amount_g_or_ml_per_serving, n.calories_per_100g_or_serving
        FROM recipe_ingredients ri
        JOIN nutrition n ON ri.ingredient_id = n.ingredient_id
        WHERE ri.recipe_id=?
        """,
        (recipe_id,),
    )
    per_person_total = 0.0
    for amount, calories_per_100g in cur.fetchall():
        if amount is None or calories_per_100g is None:
            continue
        per_person_total += (amount * calories_per_100g) / 100.0
    cur.execute("SELECT servings FROM recipes WHERE recipe_id=?", (recipe_id,))
    row = cur.fetchone()
    servings = row[0] if row else 1
    cur.execute(
        "UPDATE recipes SET per_person_calories=?, total_calories=? WHERE recipe_id=?",
        (per_person_total, per_person_total * (servings or 1), recipe_id),
    )


def upsert_images(rows: Iterable[Dict[str, Any]], db_path: Path = DEFAULT_DB) -> List[Dict[str, Any]]:
    saved: List[Dict[str, Any]] = []
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        for row in rows:
            image_id = row.get("image_id")
            payload = {
                "recipe_id": row.get("recipe_id"),
                "image_url_or_blob_ref": row.get("image_url_or_blob_ref"),
                "style_notes": row.get("style_notes"),
                "generation_parameters": json.dumps(row.get("generation_parameters")) if isinstance(row.get("generation_parameters"), (dict, list)) else row.get("generation_parameters"),
            }
            if image_id:
                cur.execute(
                    """
                    UPDATE images
                    SET recipe_id=:recipe_id, image_url_or_blob_ref=:image_url_or_blob_ref,
                        style_notes=:style_notes, generation_parameters=:generation_parameters
                    WHERE image_id=:image_id
                    """,
                    {**payload, "image_id": image_id},
                )
            else:
                cur.execute(
                    """
                    INSERT INTO images (recipe_id, image_url_or_blob_ref, style_notes, generation_parameters)
                    VALUES (:recipe_id, :image_url_or_blob_ref, :style_notes, :generation_parameters)
                    """,
                    payload,
                )
                image_id = cur.lastrowid
            cur.execute("SELECT * FROM images WHERE image_id=?", (image_id,))
            saved.append(dict(cur.fetchone()))
    return saved


def fetch_tables(db_path: Path = DEFAULT_DB) -> Dict[str, List[Dict[str, Any]]]:
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM ingredients")
        ingredients = rows_to_dicts(cur)
        cur.execute("SELECT * FROM nutrition")
        nutrition = rows_to_dicts(cur)
        cur.execute("SELECT * FROM recipes")
        recipes_raw = rows_to_dicts(cur)
        cur.execute("SELECT * FROM recipe_ingredients")
        recipe_ingredients = rows_to_dicts(cur)
        cur.execute("SELECT * FROM images")
        images = rows_to_dicts(cur)
    return {
        "ingredients": ingredients,
        "nutrition": nutrition,
        "recipes": recipes_raw,
        "recipe_ingredients": recipe_ingredients,
        "images": images,
    }


def get_state(key: str, db_path: Path = DEFAULT_DB) -> Optional[str]:
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT value FROM session_state WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None


def set_state(key: str, value: str, db_path: Path = DEFAULT_DB) -> None:
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO session_state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


if __name__ == "__main__":
    init_db()
    print("Initialized", DEFAULT_DB)
