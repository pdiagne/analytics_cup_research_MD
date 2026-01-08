# SkillCorner X PySport Analytics Cup
This repository contains the submission template for the SkillCorner X PySport Analytics Cup **Research Track**. 


## Research Track Abstract Template (max. 500 words)
#### Introduction
This study develops a machine learning framework to quantify the offensive value of off-ball movements in soccer by predicting expected threat (xThreat) from spatiotemporal tracking data. The feature set is derived from geometric relationships between attacking players, defenders, and goal positions during off-ball runs (OBR), passing options (PO), and on ball engagement (OBE). The figure below shows the geometric relationships for each event. 

[Pitch Figures_All.pdf](https://github.com/user-attachments/files/24492233/Pitch.Figures_All.pdf)

#### Methods
Data used in the analysis includes tracking and event data from 10 professional matches using SkillCorner's open dataset. The 16 engineered features extracted from this data capture the three key tactical situations: OBR (distance and angular relationships between attackers, defenders, and goal), PO (geometric configurations between ball carrier and potential receivers), and OBE (pressure situations on the ball carrier). The OBR and PO tactical situations are applied to all other attacking players in addition to the player targeted for the pass (excluding the goal keeper). All calculations are based only on the end frame of the event. An XGBoost regression model with hyperparameter tuning (n_estimators=600, max_depth=10, learning_rate=0.01) using 5-fold cross-validation was used to forecast xthreat. The model was trained on 9 matches and tested on a held-out match.

#### Results
Feature importance analysis identified the most predictive spatial relationships for xThreat generation. The figure below includes the training and testing results of match_id  = 1886347. The file Pitch Figures and Results Figures.pdf includes results for all 10 matches.

[Results Sample.pdf](https://github.com/user-attachments/files/24492238/Results.Sample.pdf)

#### Conclusion
The model moderately quantifies the offensive value of off-ball movements using interpretable geometric features and machine learning. The model's modest performance across multiple matches suggests that spatiotemporal relationships between players may predict threat generation. This approach could provide coaches and analysts with actionable insights into how specific movement patterns create scoring opportunities. Future work could 
include:
    • Find local min and max xthreat by varying L's to determine best and worst intercept point P for a Defender.
    • Freeze all metrics but vary da (OBR and PO) and vary db (OBE) to see the effect on xthreat by moving the play closer or further away from the goal.
    • Unique models for OBR and PO. Currently combining both datasets.
    • Inlcude OBR and PO events between Player Possesion start and end frames. Currently only considering events at the start and end frame.
    • Unique models for each team. 
    • Explore alternate ML models.
    • Deep dive into feature importance.
