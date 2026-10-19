"""Educational Gradio demo; NOT a loan eligibility or lending decision."""
from pathlib import Path
import joblib
import pandas as pd
import gradio as gr

MODEL_PATH = Path(__file__).parent/"artifacts"/"credit_score_model.joblib"
if not MODEL_PATH.exists():
    raise SystemExit("Model missing. Run: python train.py --data data/train.csv")
bundle = joblib.load(MODEL_PATH)  # Load only artifacts you generated and trust.

def predict(age, income, salary, loans, history, debt, utilisation):
    values = [age, income, salary, loans, history, debt, utilisation]
    if any(v is None for v in values):
        return "Please complete all seven fields."
    if not 18 <= age <= 100 or min(income, salary, loans, history, debt) < 0:
        return "Please enter valid non-negative values and an age of 18–100."
    if not 0 <= utilisation <= 100:
        return "Credit utilisation must be between 0% and 100%."
    row = dict(bundle["defaults"])
    row.update(dict(zip(bundle["inputs"], values)))
    X = pd.DataFrame([row], columns=bundle["columns"])
    label = {0:"Poor", 1:"Standard", 2:"Good"}
    prediction = int(bundle["pipeline"].predict(X)[0])
    return (f"Estimated credit-score category: {label[prediction]}\n\n"
            "Educational demonstration only. This is NOT a loan approval, "
            "eligibility determination, credit report, or financial advice. "
            "Unentered features use training-set defaults.")

with gr.Blocks(title="Credit Score Classification Demo") as demo:
    gr.Markdown("# Credit Score Classification Demo")
    gr.Markdown(
        "A portfolio prototype based on an MSc Predictive Analytics dissertation. "
        "**No names, identity numbers or personal records are requested.** "
        "This predicts a dataset credit-score label, not loan approval."
    )
    with gr.Row():
        age = gr.Slider(18, 100, value=35, step=1, label="Age (years)")
        income = gr.Number(value=50000, label="Annual income (dataset units)")
        salary = gr.Number(value=4000, label="Monthly in-hand salary (dataset units)")
    with gr.Row():
        loans = gr.Slider(0, 20, value=2, step=1, label="Existing loans")
        history = gr.Slider(0, 600, value=120, step=1, label="Credit history (months)")
    with gr.Row():
        debt = gr.Number(value=1000, label="Outstanding debt (dataset units)")
        utilisation = gr.Slider(0, 100, value=30, step=0.1,
                                label="Credit utilisation (%)")
    result = gr.Textbox(label="Educational classification", lines=5)
    gr.Button("Estimate credit-score category").click(
        predict, [age, income, salary, loans, history, debt, utilisation], result
    )
    gr.Markdown("**Limitations:** Unspecified features use fixed training-set defaults; "
                "the source dataset may not reflect real applicants or lending practices. "
                "Do not enter sensitive or real financial information.")

if __name__ == "__main__":
    import os

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        share=False
    )
