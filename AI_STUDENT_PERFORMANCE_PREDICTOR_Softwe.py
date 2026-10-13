# -*- coding: utf-8 -*-
"""

=====================================================================
 AI STUDENT PERFORMANCE PREDICTOR
 Tkinter (GUI)  +  scikit-learn (Random Forest)  +  NumPy / Pandas
=====================================================================
 A real-life early-warning desktop app.

 It learns patterns from 1,500 student records (study habits,
 attendance, assignment scores, previous GPA, sleep) and then
 predicts whether a NEW student is likely to PASS or FAIL, along
 with a confidence score and personalised recommendations.

 Run:      python student_ai.py
 Requires: numpy, pandas, scikit-learn   (tkinter ships with Python)
=====================================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------------------------
#  THEME  (dark "dashboard" palette)
# ---------------------------------------------------------------------
BG      = "#0f172a"   # app background
PANEL   = "#1e293b"   # card background
BORDER  = "#334155"   # card border
TEXT    = "#e2e8f0"   # primary text
MUTED   = "#94a3b8"   # secondary text
ACCENT  = "#38bdf8"   # cyan accent
PASS_C  = "#22c55e"   # green
FAIL_C  = "#f43f5e"   # red

FEATURES = ["StudyHours", "Attendance", "Assignment", "PrevGPA", "SleepHours"]

NICE_NAME = {
    "StudyHours": "Study Hours",
    "Attendance": "Attendance",
    "Assignment": "Assignment",
    "PrevGPA":    "Previous GPA",
    "SleepHours": "Sleep Hours",
}


# =====================================================================
#  1.  BUILD A DATASET
# =====================================================================
def build_dataset(n=1500, seed=42):
    """
    Generates a realistic synthetic student dataset.
    A hidden 'true' formula (with noise) decides Pass/Fail, so the
    ML model has genuine, learnable patterns to discover.
    """
    rng = np.random.default_rng(seed)

    study  = np.clip(rng.normal(5.0, 2.1, n), 0, 12)      # hrs/day
    attend = np.clip(rng.normal(78, 12, n), 40, 100)      # %
    assign = np.clip(rng.normal(68, 16, n), 0, 100)       # %
    gpa    = np.clip(rng.normal(6.8, 1.5, n), 0, 10)      # 0-10 scale
    sleep  = np.clip(rng.normal(6.9, 1.3, n), 3, 10)      # hrs/night

    # normalise to 0-100
    study_n = study / 12.0 * 100.0
    gpa_n   = gpa / 10.0 * 100.0
    sleep_n = np.clip(sleep, 0, 8) / 8.0 * 100.0

    # ---- hidden ground-truth "risk score" ----
    score = (0.30 * study_n +
             0.26 * attend +
             0.22 * assign +
             0.17 * gpa_n +
             0.05 * sleep_n)

    score -= np.where(attend < 60, 9.0, 0.0)               # poor attendance penalty
    score -= np.where(study < 2, 6.0, 0.0)                 # almost no study penalty
    score += np.where((study > 6.5) & (assign > 78), 4.0, 0.0)  # strong-work bonus

    final = score + rng.normal(0, 4.5, n)                  # real-world noise
    result = np.where(final >= 62, "Pass", "Fail")

    return pd.DataFrame({
        "StudyHours": np.round(study, 2),
        "Attendance": np.round(attend, 1),
        "Assignment": np.round(assign, 1),
        "PrevGPA":    np.round(gpa, 2),
        "SleepHours": np.round(sleep, 2),
        "Result":     result,
    })


# =====================================================================
#  2.  THE APPLICATION
# =====================================================================
class StudentPredictorApp:

    # slider specification: key -> (label, min, max, resolution, default, format)
    SPECS = [
        ("StudyHours", "Study hours per day",   0,  12,  0.5, 5.0, "{:.1f} h"),
        ("Attendance", "Attendance (%)",       40, 100,  1.0, 78.0, "{:.0f} %"),
        ("Assignment", "Assignment score (%)",  0, 100,  1.0, 68.0, "{:.0f} %"),
        ("PrevGPA",    "Previous GPA (0-10)",   0,  10,  0.1, 6.8, "{:.1f}"),
        ("SleepHours", "Sleep hours per night", 3,  10,  0.5, 7.0, "{:.1f} h"),
    ]

    def __init__(self, root):
        self.root = root
        root.title("AI Student Performance Predictor")
        root.geometry("1020x740")
        root.minsize(960, 700)
        root.configure(bg=BG)

        self.vars = {}
        self.val_labels = {}

        self._setup_style()
        self._train_model()      # AI part
        self._build_ui()         # GUI part

    # -----------------------------------------------------------------
    #  ttk styling
    # -----------------------------------------------------------------
    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Cyan slider handle on a slate trough
        style.configure(
            "Horizontal.TScale",
            troughcolor="#3b4a63",
            background=ACCENT,
            bordercolor="#3b4a63",
            lightcolor=ACCENT,
            darkcolor="#0ea5e9",
        )

    # -----------------------------------------------------------------
    #  AI: train the model
    # -----------------------------------------------------------------
    def _train_model(self):
        df = build_dataset()

        X = df[FEATURES]
        y = df["Result"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        self.dataset_size = len(df)
        self.accuracy = accuracy_score(y_test, y_pred)

        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        self.metrics = {
            "Accuracy":  self.accuracy,
            "Precision": rep["weighted avg"]["precision"],
            "Recall":    rep["weighted avg"]["recall"],
            "F1 Score":  rep["weighted avg"]["f1-score"],
        }

        self.importances = dict(zip(FEATURES, self.model.feature_importances_))

    # -----------------------------------------------------------------
    #  GUI
    # -----------------------------------------------------------------
    def _build_ui(self):
        # ---------- HEADER ----------
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=26, pady=(18, 8))

        tk.Label(header, text="🎓  Student Performance Predictor",
                 bg=BG, fg=TEXT,
                 font=("Segoe UI Semibold", 20)).pack(anchor="w")

        tk.Label(header,
                 text=(f"Random Forest AI  •  trained on {self.dataset_size:,} student records "
                       f"•  test accuracy {self.accuracy*100:.1f}%"),
                 bg=BG, fg=MUTED, font=("Segoe UI", 10)).pack(anchor="w", pady=(3, 0))

        # ---------- BODY ----------
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=26, pady=(6, 20))
        body.grid_columnconfigure(0, weight=0, minsize=400)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self._build_input_panel(body)
        self._build_right_panel(body)

    # -----------------------------------------------------------------
    def _build_input_panel(self, parent):
        card = tk.Frame(parent, bg=PANEL, highlightthickness=1,
                        highlightbackground=BORDER)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        tk.Label(card, text="STUDENT PROFILE", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=18, pady=(16, 0))

        tk.Label(card, text="Adjust the sliders to describe the student",
                 bg=PANEL, fg=TEXT, font=("Segoe UI", 11)).pack(anchor="w", padx=18, pady=(2, 8))

        # --- buttons pinned to the bottom ---
        btns = tk.Frame(card, bg=PANEL)
        btns.pack(side="bottom", fill="x", padx=18, pady=18)

        predict_btn = tk.Button(
            btns, text="⚡  PREDICT RESULT",
            bg=ACCENT, fg="#08111f",
            activebackground="#0ea5e9", activeforeground="#08111f",
            relief="flat", bd=0, cursor="hand2",
            font=("Segoe UI Semibold", 11),
            padx=10, pady=11,
            command=self.predict,
        )
        predict_btn.pack(side="left", fill="x", expand=True)

        reset_btn = tk.Button(
            btns, text="Reset",
            bg="#334155", fg=TEXT,
            activebackground="#475569", activeforeground=TEXT,
            relief="flat", bd=0, cursor="hand2",
            font=("Segoe UI Semibold", 11),
            padx=14, pady=11,
            command=self.reset,
        )
        reset_btn.pack(side="left", padx=(10, 0))

        # --- sliders ---
        sliders = tk.Frame(card, bg=PANEL)
        sliders.pack(fill="both", expand=True, padx=0, pady=(4, 0))
        sliders.grid_columnconfigure(0, weight=1)

        row = 0
        for key, label, lo, hi, res, default, fmt in self.SPECS:
            tk.Label(sliders, text=label, bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 10)).grid(
                row=row, column=0, sticky="w", padx=(18, 0), pady=(14, 0))

            value_lbl = tk.Label(sliders, text=fmt.format(default), bg=PANEL,
                                 fg=ACCENT, font=("Segoe UI Semibold", 10))
            value_lbl.grid(row=row, column=1, sticky="e", padx=(0, 18), pady=(14, 0))

            var = tk.DoubleVar(value=default)
            self.vars[key] = var
            self.val_labels[key] = value_lbl

            ttk.Scale(
                sliders, from_=lo, to=hi, variable=var,
                orient="horizontal",
                command=lambda v, k=key: self._on_slide(k, v),
            ).grid(row=row + 1, column=0, columnspan=2,
                   sticky="ew", padx=18, pady=(4, 0))

            row += 2

    # -----------------------------------------------------------------
    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(2, weight=1)

        # ---------------- RESULT CARD ----------------
        result_card = tk.Frame(right, bg=PANEL, highlightthickness=1,
                               highlightbackground=BORDER)
        result_card.grid(row=0, column=0, sticky="ew")

        tk.Label(result_card, text="PREDICTION", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=20, pady=(14, 0))

        self.result_lbl = tk.Label(result_card, text="Awaiting input…",
                                   bg=PANEL, fg=MUTED,
                                   font=("Segoe UI Semibold", 26))
        self.result_lbl.pack(anchor="w", padx=20)

        self.prob_lbl = tk.Label(result_card, text="Press  ⚡ PREDICT RESULT  to run the AI",
                                 bg=PANEL, fg=MUTED, font=("Segoe UI", 10))
        self.prob_lbl.pack(anchor="w", padx=20, pady=(2, 0))

        bar_wrap = tk.Frame(result_card, bg=BORDER, height=12)
        bar_wrap.pack(fill="x", padx=20, pady=(12, 18))
        bar_wrap.pack_propagate(False)

        self.bar_fill = tk.Frame(bar_wrap, bg=ACCENT)
        self.bar_fill.place(x=0, y=0, relwidth=0.0, relheight=1.0)

        # ---------------- ADVICE CARD ----------------
        advice_card = tk.Frame(right, bg=PANEL, highlightthickness=1,
                               highlightbackground=BORDER)
        advice_card.grid(row=1, column=0, sticky="ew", pady=(14, 0))

        tk.Label(advice_card, text="PERSONALISED RECOMMENDATIONS", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=20, pady=(14, 4))

        self.advice_lbl = tk.Label(
            advice_card,
            text="Recommendations will appear here after a prediction.",
            bg=PANEL, fg=TEXT, font=("Segoe UI", 10),
            justify="left", anchor="w", wraplength=520,
        )
        self.advice_lbl.pack(anchor="w", fill="x", padx=20, pady=(0, 16))

        # ---------------- INSIGHTS CARD ----------------
        insights = tk.Frame(right, bg=PANEL, highlightthickness=1,
                            highlightbackground=BORDER)
        insights.grid(row=2, column=0, sticky="nsew", pady=(14, 0))

        tk.Label(insights, text="MODEL INSIGHTS", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=20, pady=(14, 8))

        # --- metric chips ---
        chips = tk.Frame(insights, bg=PANEL)
        chips.pack(fill="x", padx=20)

        for i, (name, value) in enumerate(self.metrics.items()):
            chip = tk.Frame(chips, bg="#273449")
            chip.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 8, 0))
            chips.grid_columnconfigure(i, weight=1)

            tk.Label(chip, text=name.upper(), bg="#273449", fg=MUTED,
                     font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=10, pady=(8, 0))
            tk.Label(chip, text=f"{value*100:.1f}%", bg="#273449", fg=ACCENT,
                     font=("Segoe UI Semibold", 14)).pack(anchor="w", padx=10, pady=(0, 8))

        # --- feature importance chart ---
        tk.Label(insights, text="What drives the prediction?", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w", padx=20, pady=(14, 2))

        self.chart = tk.Canvas(insights, bg=PANEL, highlightthickness=0, height=140)
        self.chart.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.chart.bind("<Configure>", lambda e: self._draw_chart())

    # -----------------------------------------------------------------
    #  Slider callback
    # -----------------------------------------------------------------
    def _on_slide(self, key, raw):
        spec = next(s for s in self.SPECS if s[0] == key)
        _, _, _, _, res, _, fmt = spec
        snapped = round(float(raw) / res) * res
        self.val_labels[key].config(text=fmt.format(snapped))

    # -----------------------------------------------------------------
    #  Feature-importance bar chart (pure Tkinter Canvas)
    # -----------------------------------------------------------------
    def _draw_chart(self):
        c = self.chart
        c.delete("all")

        w = c.winfo_width()
        h = c.winfo_height()
        if w < 50 or h < 30:
            return

        items = sorted(self.importances.items(), key=lambda kv: kv[1], reverse=True)
        max_v = max(v for _, v in items) or 1.0

        label_w = 110
        bar_area = w - label_w - 62
        bar_h = 18
        gap = (h - len(items) * bar_h) / (len(items) + 1)

        y = gap
        for key, value in items:
            c.create_text(label_w - 8, y + bar_h / 2,
                          text=NICE_NAME[key], anchor="e",
                          fill=MUTED, font=("Segoe UI", 9))

            c.create_rectangle(label_w, y, label_w + bar_area, y + bar_h,
                               fill="#273449", outline="")

            bw = bar_area * (value / max_v)
            c.create_rectangle(label_w, y, label_w + bw, y + bar_h,
                               fill=ACCENT, outline="")

            c.create_text(label_w + bw + 8, y + bar_h / 2,
                          text=f"{value*100:.1f}%", anchor="w",
                          fill=TEXT, font=("Segoe UI Semibold", 9))

            y += bar_h + gap

    # -----------------------------------------------------------------
    #  PREDICTION
    # -----------------------------------------------------------------
    def predict(self):
        # 1) read + snap slider values
        values = {}
        for key, _, _, _, res, _, _ in self.SPECS:
            values[key] = round(self.vars[key].get() / res) * res

        student = pd.DataFrame([[values[k] for k in FEATURES]], columns=FEATURES)

        # 2) AI inference
        proba = self.model.predict_proba(student)[0]
        classes = list(self.model.classes_)
        p_pass = float(proba[classes.index("Pass")])
        prediction = "Pass" if p_pass >= 0.5 else "Fail"

        # 3) update UI
        if prediction == "Pass":
            self.result_lbl.config(text="✅  LIKELY TO PASS", fg=PASS_C)
            self.bar_fill.config(bg=PASS_C)
        else:
            self.result_lbl.config(text="⚠️  AT RISK OF FAILING", fg=FAIL_C)
            self.bar_fill.config(bg=FAIL_C)

        self.prob_lbl.config(
            text=f"Chance of passing: {p_pass*100:.1f}%   •   "
                 f"Model confidence: {max(p_pass, 1-p_pass)*100:.1f}%"
        )
        self.bar_fill.place_configure(relwidth=max(0.0, min(1.0, p_pass)))

        # 4) personalised advice
        self.advice_lbl.config(text=self._build_advice(values, prediction, p_pass))

    # -----------------------------------------------------------------
    def _build_advice(self, v, prediction, p_pass):
        tips = []

        if v["Attendance"] < 75:
            tips.append(f"• Attendance is {v['Attendance']:.0f}% — below the 75% "
                        f"threshold. This is the single biggest risk factor.")
        if v["StudyHours"] < 4:
            tips.append(f"• Only {v['StudyHours']:.1f} h of study per day. "
                        f"Push it toward 4–5 focused hours.")
        if v["Assignment"] < 65:
            tips.append(f"• Assignment score is {v['Assignment']:.0f}%. "
                        f"Submitting consistently (target 70+) lifts the odds sharply.")
        if v["PrevGPA"] < 6:
            tips.append(f"• Previous GPA {v['PrevGPA']:.1f}/10 suggests gaps in "
                        f"fundamentals — consider weekly revision sessions.")
        if v["SleepHours"] < 6:
            tips.append(f"• Sleeping {v['SleepHours']:.1f} h/night hurts memory "
                        f"consolidation. Aim for 7–8 hours.")
        if v["SleepHours"] > 9:
            tips.append("• Sleeping more than 9 hours may indicate irregular routine "
                        "— a consistent schedule works better.")

        if not tips:
            if prediction == "Pass":
                tips.append("• Excellent profile! Keep the current routine going "
                            "and maintain consistent revision before exams.")
            else:
                tips.append("• Metrics look reasonable but the combination is still "
                            "borderline — focus on attendance and assignment quality.")

        header = ("Priority actions:" if prediction == "Fail"
                  else "Keep it up — small tuning suggestions:")

        return header + "\n" + "\n".join(tips)

    # -----------------------------------------------------------------
    def reset(self):
        for key, _, _, _, _, default, fmt in self.SPECS:
            self.vars[key].set(default)
            self.val_labels[key].config(text=fmt.format(default))

        self.result_lbl.config(text="Awaiting input…", fg=MUTED)
        self.prob_lbl.config(text="Press  ⚡ PREDICT RESULT  to run the AI")
        self.bar_fill.place_configure(relwidth=0.0)
        self.advice_lbl.config(
            text="Recommendations will appear here after a prediction.")


# =====================================================================
#  3.  ENTRY POINT
# =====================================================================
def main():
    root = tk.Tk()
    StudentPredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
