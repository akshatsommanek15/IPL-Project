IPL Win Probability Predictor (2026)
An end-to-end Machine Learning web application that predicts the win probability of the chasing team in an IPL match. This project goes beyond basic run-rate calculations by incorporating match pressure and batting depth, offering a more nuanced "cricketing" perspective on game outcomes.

🚀 Live Demo
https://ipl-smart-predictor26.streamlit.app/

🧐 The Design Engineering Approach
In cricket, the scoreboard often hides the true tension of the match. A team needing 7 runs an over might seem comfortable, but the loss of key specialist batsmen changes the psychological and statistical landscape.


Core Innovations:

Wicket Pressure Logic: The model is programmed to recognize "The Wicket Cliff"—the point where losing one more batsman causes a disproportionate drop in win probability.

Tail-Ender Detection: Differentiates between a set middle-order partnership and a tail-end struggle.

Volatile Probability: Uses a Calibrated Random Forest Classifier to reflect the high-stakes, fast-changing nature of T20 cricket.


📊 Model & Performance
Algorithm: Random Forest Classifier (Calibrated)

Accuracy: ~78.79%

Training Data: 15+ years of historical IPL ball-by-ball data.

Persistence: Model serialized using dill to maintain complex pipeline structures.


🛠️ Tech Stack
Language: Python 3.10

Frontend: Streamlit

Machine Learning: Scikit-learn (v1.7.2), Dill, Pandas, NumPy

Deployment: Streamlit Community Cloud

Version Control: Git & GitHub