# AURA • Market Intelligence Terminal

A production-grade, full-stack quantitative finance application built to predict short-term equity trajectories using machine learning and Wall Street-grade technical analysis.

**[Live Dashboard Deployment]** 
(https://aura-market-terminal.streamlit.app/)

---

## 📸 Phase I: The Initial Build & Error Handling
We started by testing the initial Streamlit engine and troubleshooting syntax errors, such as the `unsafe_allow_html` keyword mismatch, to get the UI rendering correctly.

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/df188a4f-ac41-4c11-b56c-ffd5fa832cea" />

## 📸 Phase II: The Extrapolation Trap (Random Forest)
During initial testing with the Random Forest model, the AI's prediction flatlined because tree-based models cannot extrapolate beyond historical highs.

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/991d8940-81f5-42b7-b64f-567ef29ed17e" />

## 📸 Phase III: The Algorithmic Pivot (Ridge Regression)
We swapped the core engine to a Ridge Regression model. The AI successfully tracked the momentum vector, dropping the RMSE and hugging the actual spot price trajectory perfectly.

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/2bb90793-19d8-432c-8d10-60c0f34ad53e" />

## 📸 Phase IV: The UI Upgrade (Candlesticks & Metrics)
We replaced the basic line charts with professional Japanese Candlestick (OHLC) charts and expanded the metric cards to include daily opening and closing data.

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/a90f873b-833d-4605-bfd9-f08b6d818fdf" />

## 📸 Phase V: The AI Executive Agent
We integrated the Algorithmic Analyst Agent at the bottom of the terminal to dynamically generate bespoke, human-readable market briefings based on the active telemetry.

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/23e2c36e-3a12-4b82-ab91-ba864d3924d3" />

## 📸 Phase VI: Adaptive Theming & Layout Fixes
We enforced strict column symmetry `[1, 1, 1, 1]` for the metric cards and refactored the CSS to use native variables, ensuring the dashboard renders flawlessly in both Light and Dark modes.

<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/f3898a79-ea5f-4d3e-b4fd-6f132d8891ae" />

---

## 🛠️ Local Installation

To run the AURA Terminal on your local machine:

```bash
git clone [https://github.com/yourusername/aura-market-terminal.git](https://github.com/yourusername/aura-market-terminal.git)
cd aura-market-terminal
pip install -r requirements.txt
streamlit run app.py
