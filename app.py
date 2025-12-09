import json
from typing import List
import pandas as pd

import streamlit as st

from db import DEFAULT_DB, fetch_tables, init_db
from workflow import KitchenCrew, encode_uploaded_images, IMAGE_STORE

# Reset DB each run so tables start fresh
if DEFAULT_DB.exists():
    DEFAULT_DB.unlink()
init_db(DEFAULT_DB)


st.set_page_config(page_title="Agentic Kitchen Planner", page_icon="APP", layout="wide")


def get_crew() -> KitchenCrew:
    if "crew_instance" not in st.session_state:
        st.session_state.crew_instance = KitchenCrew()
    return st.session_state.crew_instance


crew = get_crew()
st.title("Agentic Kitchen Planner")
st.caption("Agentic workflow that goes from pantry photos to recipes + images.")

with st.sidebar:
    st.markdown("Use OPENAI_API_KEY in your environment. Models default to gpt-4o-mini.")
    hints = st.text_area("Optional hints", placeholder="Ignore canned drinks; focus on proteins and veggies.")
    meal_time = st.selectbox("Meal time", ["breakfast", "lunch", "dinner"])
    target_calories = st.number_input("Target calories per person", min_value=200, max_value=1500, value=650, step=50)
    num_people = st.number_input("Number of people", min_value=1, max_value=12, value=2, step=1)
    constraints = st.text_input("Constraints", placeholder="High protein, must use spinach, etc.")

uploaded_files = st.file_uploader(
    "Upload fridge/pantry/counter images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

run = st.button("Run full workflow", type="primary")

if run:
    images_payload: List[dict] = []
    if uploaded_files:
        images_payload = encode_uploaded_images(
            [f.getvalue() for f in uploaded_files], [f.name for f in uploaded_files]
        )
    with st.spinner("Running agentic workflow..."):
        output = crew.run_primary_flow(
            images=images_payload,
            hints=hints,
            meal_time=meal_time,
            target_calories_per_person=target_calories,
            num_people=int(num_people),
            constraints=constraints,
        )
    st.success("Workflow complete")
    st.subheader("Task outputs")
    st.json(output.get("task_outputs"))
    st.subheader("Crew result")
    st.write(output.get("result"))

    # --- Recipe cards + images ---
    task_outputs = output.get("task_outputs", {})
    compose_output = task_outputs.get("ComposeRecipesAndImagesForDisplay")
    cards_obj = None

    if compose_output:
        # CrewAI Task.output is usually a TaskOutput with a `.raw` JSON string
        raw = getattr(compose_output, "raw", compose_output)

        if isinstance(raw, str):
            try:
                cards_obj = json.loads(raw)
            except Exception:
                cards_obj = None
        elif isinstance(raw, dict):
            cards_obj = raw

    print("Recipe cards object found: ", cards_obj is not None)
    if isinstance(cards_obj, dict) and "recipe_cards" in cards_obj:
        print("Processing recipe cards object.")

        st.subheader("Recipe cards")
        for card in cards_obj["recipe_cards"]:
            st.markdown(f"### {card['name']}")
            st.write(card.get("short_description", ""))

            img_ref = card.get("image_url_or_blob_ref")
            if not img_ref:
                continue

            # 1) stub case
            if isinstance(img_ref, str) and img_ref.startswith("stub://"):
                st.image(
                    "assets/placeholder_bowl.png",
                    caption=card["name"],
                    use_column_width=True,
                )

            # 2) generated handle -> IMAGE_STORE bytes
            elif isinstance(img_ref, str) and img_ref.startswith("gen://"):
                key = img_ref[len("gen://") :]
                img_bytes = IMAGE_STORE.get(key)
                if img_bytes:
                    st.image(img_bytes, caption=card["name"], use_column_width=True)
                else:
                    st.image(
                        "assets/placeholder_bowl.png",
                        caption=card["name"],
                        use_column_width=True,
                    )

            # 3) actual URL or local path
            else:
                st.image(img_ref, caption=card["name"], use_column_width=True)



st.divider()
st.subheader("Current tables")

tables = crew.fetch_tables()
for name, rows in tables.items():
    st.markdown(f"**{name}**")
    if rows:
        # Normalize to a pandas DataFrame
        if isinstance(rows, pd.DataFrame):
            df = rows.copy()
        else:
            df = pd.DataFrame(rows)

        # Coerce all columns to string so PyArrow doesn't choke on mixed types
        df = df.astype("string")

        # New API: use width instead of use_container_width
        st.dataframe(df, width="stretch")
    else:
        st.write("(empty)")
