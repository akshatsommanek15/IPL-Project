import streamlit as st
import pandas as pd
import numpy as np
import dill
import os

# --- 1. PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")

# --- 2. MODEL LOADING ---
model_path = os.path.join('artifacts', 'best_model.pkl')

try:
    with open(model_path, 'rb') as f:
        model = dill.load(f)
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Please check your folder structure on GitHub!")
    st.stop() 

# --- 3. TABS SETUP ---
tab1, tab2 = st.tabs(["🏏 Win Predictor", "📊 Model Performance"])

# --- TAB 1: PREDICTOR ---
with tab1:
    st.title("IPL Win Probability Predictor 🏏")
    st.markdown("---")

    teams = [
        'Mumbai Indians', 'Kolkata Knight Riders', 'Rajasthan Royals',
        'Royal Challengers Bengaluru', 'Chennai Super Kings',
        'Sunrisers Hyderabad', 'Delhi Capitals', 'Punjab Kings',
        'Lucknow Super Giants', 'Gujarat Titans'
    ]

    with st.sidebar:
        st.header("Match Setup")
        batting_team = st.selectbox('Batting Team', sorted(teams))
        bowling_team = st.selectbox('Bowling Team', sorted(teams), index=1)
        
        toss_winner = st.selectbox('Toss Winner', [batting_team, bowling_team])
        toss_decision = st.selectbox('Toss Decision', ['bat', 'field'])
        
        target = st.number_input('Target Score', min_value=1, value=180)

    # --- Main Input Columns ---
    col1, col2, col3 = st.columns(3)

    with col1:
        current_score = st.number_input('Current Score', min_value=0)
    with col2:
        overs_done = st.number_input('Overs Done (e.g., 10.3)', min_value=0.0, max_value=19.5, step=0.1, format="%.1f")
    with col3:
        wickets_out = st.number_input('Wickets Fallen', min_value=0, max_value=9)

    if st.button('Predict Win Probability'):
        # --- Calculations ---
        overs_int = int(overs_done)
        extra_balls = round((overs_done - overs_int) * 10)
        total_balls_bowled = (overs_int * 6) + extra_balls
        
        if total_balls_bowled > 120: total_balls_bowled = 120
        
        balls_left = 120 - total_balls_bowled
        runs_left = target - current_score
        wickets_left = 10 - wickets_out
        overs_completed_math = total_balls_bowled / 6
        
        # Feature Engineering
        crr = current_score / overs_completed_math if overs_completed_math > 0 else 0
        toss_advantage = 1 if toss_winner == batting_team else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
        is_tail_batting = 1 if wickets_out >= 6 else 0
        over_left = balls_left / 6
        wicket_pressure = wickets_out / (overs_completed_math + 1)
        momentum = crr - rrr 
        runs_per_wicket = runs_left / (wickets_left + 1)
        balls_per_wicket = balls_left / (wickets_left + 1)
        
        if total_balls_bowled <= 36:
            phase = 'powerplay'
        elif total_balls_bowled <= 90:
            phase = 'middle_overs'
        else:
            phase = 'death_overs'

        # Construct DataFrame
        input_df = pd.DataFrame({
            'runs_target': [target],
            'balls_left': [balls_left],
            'runs_left': [runs_left],
            'wickets_left': [wickets_left],
            'current_run_rate': [crr],
            'toss_advantage': [toss_advantage],
            'required_run_rate': [rrr],
            'is_tail_batting': [is_tail_batting],
            'over_left': [over_left],
            'wickets_fallen': [wickets_out],
            'wicket_pressure': [wicket_pressure],
            'momentum': [momentum],
            'runs_per_wicket': [runs_per_wicket],
            'balls_per_wicket': [balls_per_wicket],
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'toss_decision': [toss_decision],
            'phase': [phase]
        })

        # Prediction
        result = model.predict_proba(input_df)
        prob_loss = result[0][0]
        prob_win = result[0][1]

        # Results Display
        st.markdown("### 📊 Match Statistics")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        overs_rem = balls_left // 6
        balls_rem = balls_left % 6
        overs_left_str = f"{overs_rem}.{balls_rem}"

        with stat_col1:
            st.metric(label="Runs Needed", value=runs_left)
        with stat_col2:
             st.metric(label="Overs Left", value=overs_left_str)
        with stat_col3:
            st.metric(label="Required RR", value=round(rrr, 2), delta=round(rrr - crr, 2), delta_color="inverse")
        with stat_col4:
            st.metric(label="Current RR", value=round(crr, 2))

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader(f"{batting_team}")
            st.title(f"{round(prob_win * 100)}%")
        with res_col2:
            st.subheader(f"{bowling_team}")
            st.title(f"{round(prob_loss * 100)}%")
            
        st.progress(prob_win)

# --- TAB 2: PERFORMANCE ---
with tab2:
    st.header("Technical Model Evaluation")
    st.write("""
    To ensure the highest reliability, this **Random Forest** model was trained on 
    15+ years of historical IPL data, filtered to the current 10 active teams.
    """)
    
    report_path = os.path.join('artifacts', 'model_report.png')
    
    if os.path.exists(report_path):
        st.image(report_path, caption="Model Classification Report (Accuracy: 78.79%)", width='stretch')
    else:
        st.warning("Model report image not found in the artifacts folder.")

    st.write("### Key Takeaways:")
    st.write("- **Precision:** High accuracy in predicting chasing team wins (~81%).")
    st.write("- **F1-Score:** Balanced performance across both defending and chasing scenarios.")