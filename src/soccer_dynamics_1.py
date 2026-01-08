
def train_xgb_model(X, Y, M_train, test_size=0.2, random_state=42):
    """
    Train and tune an XGBoost regression model with cross-validation and
    feature importance analysis.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    Y : pd.Series or np.ndarray
        Target vector.
    M_train : pd.DataFrame
        Original dataset including feature names (for interpretability).
    test_size : float, optional
        Fraction of data to reserve for validation (default=0.2).
    random_state : int, optional
        Random seed for reproducibility (default=42).

    Returns
    -------
    dict
        Dictionary containing the trained model, metrics, predictions,
        and feature importance DataFrame.
    """
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    import pandas as pd
    import xgboost as xgb

    # Ensure proper target shape
    if len(Y.shape) > 1 and Y.shape[1] == 1:
        Y = Y.ravel()

    # Split into train/validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # Feature names
    target_cols = ['xthreat']  # or detect automatically
    feature_cols = [col for col in M_train.columns if col not in target_cols]
    feature_names = feature_cols

    # Base model (for initial training and CV)
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1
    )

    # Fit base model
    xgb_model.fit(X_train, y_train)

    # Base predictions
    y_pred = xgb_model.predict(X_val)

    # Evaluate base model
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    # Cross-validation R² scores
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')

    # Hyperparameter tuning grid
    param_grid = {
        'n_estimators': [600],
        'max_depth': [10],
        'learning_rate': [0.01],
    }

    xgb_base = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=random_state,
        n_jobs=-1
    )

    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Best tuned model
    best_xgb_model = grid_search.best_estimator_
    y_pred_best = best_xgb_model.predict(X_val)

    # Best model metrics
    r2_best = r2_score(y_val, y_pred_best)
    mse_best = mean_squared_error(y_val, y_pred_best)
    rmse_best = np.sqrt(mse_best)
    mae_best = mean_absolute_error(y_val, y_pred_best)

    # Feature importance
    feature_importance = best_xgb_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    # Save best model and feature importances
    import joblib
    joblib.dump(best_xgb_model, 'xgb_xthreat_best_model.pkl')
    importance_df.to_csv('feature_importance.csv', index=False)

    return {
        'best_model': best_xgb_model,
        'importance_df': importance_df,
        'metrics': {
            'R2': r2_best,
            'MAE': mae_best,
            'RMSE': rmse_best,
            'MSE': mse_best,
            'CV_R2': cv_scores.mean()
        },
        'predictions': {
            'y_val': y_val,
            'y_pred': y_pred_best
        }
    }


def calculate_metrics(frame_type, frame_num, Lf_OBR, Lf_PO, Lf_OBE,po_obr,tracking,match_id):
    # ************
    # Off Ball Run calcs
    # ************

    Frame = frame_num  # Same for all scenarios

    # OBR attacker and PO attacker
    a_player_id = list(set(po_obr['player_id'][po_obr[frame_type] == Frame]))
    a_player_id = int(a_player_id[0])
    ind = (tracking['frame'] == Frame) & (tracking['player_id'] == a_player_id)
    x_A = float(tracking.loc[ind, 'x'].iloc[0])
    y_A = float(tracking.loc[ind, 'y'].iloc[0])
    A = Point(x_A, y_A)  # attacker (off-ball)

    # Ball carrier for OBR
    b_player_id = list(set(tracking['possession_player_id'][tracking['frame'] == Frame]))
    b_player_id = int(b_player_id[0])
    ind = (tracking['frame'] == Frame) & (tracking['player_id'] == b_player_id)
    x_B = float(tracking.loc[ind, 'x'].iloc[0])
    y_B = float(tracking.loc[ind, 'y'].iloc[0])
    B = Point(x_B, y_B)  # ball carrier

    # Find the defender that is closest to La
    d_player_ids = list(set(tracking['player_id'][(tracking['frame'] == Frame) & (tracking['possession_flag'] == 'OOP')
                                                  & (tracking['is_gk'] == False)]))
    ind = (tracking['frame'] == Frame) & tracking['player_id'].isin(d_player_ids)

    # Create defenders list dynamically from the coordinates
    coords = tracking.loc[ind, ['x', 'y']].to_numpy()
    defenders = [Point(row[0], row[1]) for row in coords]
    # There should typically always be 10 defenders (excluding GK) but this accounts for red cards

    # Goal center (attacking right to left)
    G = Point(-52.5, 0)

    # da: distance from attacker A to goal G
    da = distance(A, G)

    # Choose La (distance from A along line AG) as needed.
    # Lf_OBR = 0.3 # factor to determine La as a percentage of da
    La = Lf_OBR * da

    # Point P on line from A to G at distance La from A
    P = point_on_segment(A, G, La)

    # Find closest defender to P
    D_closest, dd, closest_idx = find_closest_defender(defenders, P)
    x_D_closest = D_closest.x
    y_D_closest = D_closest.y

    # db: distance from ball carrier B to point P
    db = distance(B, P)

    # Θa: direction from A to G
    Theta_a = angle_deg(A, G)

    # Θd: direction from closest defender to P
    Theta_d = angle_deg(D_closest, P)

    # Θb: direction from B to P
    Theta_b = angle_deg(B, P)

    # Rename measures
    # x_G_OBR = -52.5
    # y_G_OBR = 0
    Theta_a_OBR = Theta_a
    Theta_d_OBR = Theta_d
    Theta_b_OBR = Theta_b
    da_OBR = da
    dd_OBR = dd
    db_OBR = db  # ************
    # Passing option
    # ************

    # db: distance from ball carrier B to A
    db = distance(B, A)

    # Choose Lb (distance from B along line BA) as needed.
    # Lf_PO = 0.3
    Lb = Lf_PO * db

    # Point P on line from B to A at distance Lb from B
    P = point_on_segment(B, A, Lb)

    # Find closest defender to P
    D_closest, dd, closest_idx = find_closest_defender(defenders, P)

    # Θd: direction from closest defender to P
    Theta_d = angle_deg(D_closest, P)

    # Θb: direction from B to P
    Theta_b = angle_deg(B, P)

    # Rename measures
    Theta_d_PO = Theta_d
    Theta_b_PO = Theta_b
    dd_PO = dd
    db_PO = db  # ************
    # On ball engagement
    # ************

    # db: distance from B to G
    db = distance(B, G)

    # Choose Lb (distance from B along line BG).
    # Lf_OBE = 0.3 # factor to determine Lb as a percentage of db
    Lb = Lf_OBE * db

    # Point P on line from B to G at distance Lb from B
    P = point_on_segment(B, G, Lb)

    # Find closest defender to P
    D_closest_prime, dd_prime, closest_idx_prime = find_closest_defender(defenders, P)

    # Find closest defender to B
    D_closest, dd, closest_idx = find_closest_defender(defenders, B)

    # Θd_prime: direction from closest defender to P
    Theta_d_prime = angle_deg(D_closest_prime, P)

    # Θd: direction from closest defender to B
    Theta_d = angle_deg(D_closest, B)

    # Θb: direction from B to G
    Theta_b = angle_deg(B, G)

    # Rename measures
    Theta_d_OBE = Theta_d
    Theta_d_prime_OBE = Theta_d_prime
    Theta_b_OBE = Theta_b
    dd_OBE = dd
    dd_prime_OBE = dd_prime
    db_OBE = db  # ************
    # Create regression X and Y
    # ************

    # Aggregate OBR, PO and OBE calcs
    X = [Theta_a_OBR, Theta_d_OBR, Theta_b_OBR, da_OBR, dd_OBR, db_OBR,
         Theta_d_PO, Theta_b_PO, dd_PO, db_PO,
         Theta_d_OBE, Theta_d_prime_OBE, Theta_b_OBE, dd_OBE, dd_prime_OBE, db_OBE]

    ind = (po_obr[frame_type] == Frame) & (po_obr['player_id'] == a_player_id)
    Y = float(po_obr.loc[ind, 'xthreat'].iloc[0])  # Single column, first match

    # Convert to dataframe
    columns = ['xthreat', 'Theta_a_OBR', 'Theta_d_OBR', 'Theta_b_OBR',
               'da_OBR', 'dd_OBR', 'db_OBR', 'Theta_d_PO', 'Theta_b_PO', 'dd_PO', 'db_PO',
               'Theta_d_OBE', 'Theta_d_prime_OBE', 'Theta_b_OBE', 'dd_OBE', 'dd_prime_OBE', 'db_OBE']
    M = pd.DataFrame([[Y] + X], columns=columns)
    M['match_id'] = match_id

    return M


def calculate_metrics_extended(frame_type, frame_num, Lf_OBR, Lf_PO, Lf_OBE, po_obr, tracking, match_id,
                               possession_player_id, a_player_id, event_id=None):
    """
    Calculate metrics for the primary teammate (from po_obr).
    This is essentially your original calculate_metrics function with minor modifications.
    """
    # ************
    # Off Ball Run calcs
    # ************

    Frame = frame_num

    # OBR attacker and PO attacker
    ind = (tracking['frame'] == Frame) & (tracking['player_id'] == a_player_id)
    x_A = float(tracking.loc[ind, 'x'].iloc[0])
    y_A = float(tracking.loc[ind, 'y'].iloc[0])
    A = Point(x_A, y_A)  # attacker (off-ball)

    # Ball carrier for OBR
    ind = (tracking['frame'] == Frame) & (tracking['player_id'] == possession_player_id)
    x_B = float(tracking.loc[ind, 'x'].iloc[0])
    y_B = float(tracking.loc[ind, 'y'].iloc[0])
    B = Point(x_B, y_B)  # ball carrier

    # Find the defender that is closest to La
    d_player_ids = list(set(tracking['player_id'][(tracking['frame'] == Frame) & (tracking['possession_flag'] == 'OOP')
                                                  & (tracking['is_gk'] == False)]))
    ind = (tracking['frame'] == Frame) & tracking['player_id'].isin(d_player_ids)

    # Create defenders list dynamically from the coordinates
    coords = tracking.loc[ind, ['x', 'y']].to_numpy()
    defenders = [Point(row[0], row[1]) for row in coords]

    # Goal center (attacking right to left)
    G = Point(-52.5, 0)

    # da: distance from attacker A to goal G
    da = distance(A, G)

    # Choose La (distance from A along line AG) as needed.
    La = Lf_OBR * da

    # Point P on line from A to G at distance La from A
    P = point_on_segment(A, G, La)

    # Find closest defender to P
    D_closest, dd, closest_idx = find_closest_defender(defenders, P)
    x_D_closest = D_closest.x
    y_D_closest = D_closest.y

    # db: distance from ball carrier B to point P
    db = distance(B, P)

    # Θa: direction from A to G
    Theta_a = angle_deg(A, G)

    # Θd: direction from closest defender to P
    Theta_d = angle_deg(D_closest, P)

    # Θb: direction from B to P
    Theta_b = angle_deg(B, P)

    # Rename measures
    Theta_a_OBR = Theta_a
    Theta_d_OBR = Theta_d
    Theta_b_OBR = Theta_b
    da_OBR = da
    dd_OBR = dd
    db_OBR = db

    # ************
    # Passing option
    # ************

    # db: distance from ball carrier B to A
    db = distance(B, A)

    # Choose Lb (distance from B along line BA) as needed.
    Lb = Lf_PO * db

    # Point P on line from B to A at distance Lb from B
    P = point_on_segment(B, A, Lb)

    # Find closest defender to P
    D_closest, dd, closest_idx = find_closest_defender(defenders, P)

    # Θd: direction from closest defender to P
    Theta_d = angle_deg(D_closest, P)

    # Θb: direction from B to P
    Theta_b = angle_deg(B, P)

    # Rename measures
    Theta_d_PO = Theta_d
    Theta_b_PO = Theta_b
    dd_PO = dd
    db_PO = db

    # ************
    # On ball engagement
    # ************

    # db: distance from B to G
    db = distance(B, G)

    # Choose Lb (distance from B along line BG).
    Lb = Lf_OBE * db

    # Point P on line from B to G at distance Lb from B
    P = point_on_segment(B, G, Lb)

    # Find closest defender to P
    D_closest_prime, dd_prime, closest_idx_prime = find_closest_defender(defenders, P)

    # Find closest defender to B
    D_closest, dd, closest_idx = find_closest_defender(defenders, B)

    # Θd_prime: direction from closest defender to P
    Theta_d_prime = angle_deg(D_closest_prime, P)

    # Θd: direction from closest defender to B
    Theta_d = angle_deg(D_closest, B)

    # Θb: direction from B to G
    Theta_b = angle_deg(B, G)

    # Rename measures
    Theta_d_OBE = Theta_d
    Theta_d_prime_OBE = Theta_d_prime
    Theta_b_OBE = Theta_b
    dd_OBE = dd
    dd_prime_OBE = dd_prime
    db_OBE = db

    # ************
    # Create regression X and Y
    # ************

    # Aggregate OBR, PO and OBE calcs
    X = [Theta_a_OBR, Theta_d_OBR, Theta_b_OBR, da_OBR, dd_OBR, db_OBR,
         Theta_d_PO, Theta_b_PO, dd_PO, db_PO,
         Theta_d_OBE, Theta_d_prime_OBE, Theta_b_OBE, dd_OBE, dd_prime_OBE, db_OBE]

    ind = (po_obr[frame_type] == Frame) & (po_obr['player_id'] == a_player_id)
    Y = float(po_obr.loc[ind, 'xthreat'].iloc[0])  # Single column, first match

    # Convert to dataframe
    columns = ['xthreat', 'Theta_a_OBR', 'Theta_d_OBR', 'Theta_b_OBR',
               'da_OBR', 'dd_OBR', 'db_OBR', 'Theta_d_PO', 'Theta_b_PO', 'dd_PO', 'db_PO',
               'Theta_d_OBE', 'Theta_d_prime_OBE', 'Theta_b_OBE', 'dd_OBE', 'dd_prime_OBE', 'db_OBE']
    M = pd.DataFrame([[Y] + X], columns=columns)
    M['match_id'] = match_id
    M['frame'] = Frame
    M['frame_type'] = frame_type
    M['possession_player_id'] = possession_player_id
    M['teammate_player_id'] = a_player_id

    # Add event_id if provided
    if event_id is not None:
        M['event_id'] = event_id

    return M

def calculate_teammate_metrics(frame_num, Lf_OBR, Lf_PO, Lf_OBE, tracking, possession_player_id, teammate_id):
    """
    Calculate simplified metrics for additional teammates (without xthreat).
    Returns a dictionary of metric names and values.
    """
    Frame = frame_num

    # Get teammate coordinates
    teammate_data = tracking[(tracking['frame'] == Frame) & (tracking['player_id'] == teammate_id)]
    x_T = float(teammate_data['x'].iloc[0])
    y_T = float(teammate_data['y'].iloc[0])
    T = Point(x_T, y_T)

    # Get ball carrier coordinates
    ball_carrier_data = tracking[(tracking['frame'] == Frame) & (tracking['player_id'] == possession_player_id)]
    x_B = float(ball_carrier_data['x'].iloc[0])
    y_B = float(ball_carrier_data['y'].iloc[0])
    B = Point(x_B, y_B)

    # Find defenders
    defenders_data = tracking[(tracking['frame'] == Frame) &
                              (tracking['possession_flag'] == 'OOP') &
                              (tracking['is_gk'] == False)]
    coords = defenders_data[['x', 'y']].to_numpy()
    defenders = [Point(row[0], row[1]) for row in coords]

    # Goal center
    G = Point(-52.5, 0)

    # 1. Off Ball Run metrics for teammate
    da = distance(T, G)
    La = Lf_OBR * da
    P = point_on_segment(T, G, La)
    D_closest, dd, _ = find_closest_defender(defenders, P)
    db = distance(B, P)
    Theta_a = angle_deg(T, G)
    Theta_d = angle_deg(D_closest, P)
    Theta_b = angle_deg(B, P)

    # 2. Passing option metrics for teammate
    db_po = distance(B, T)
    Lb = Lf_PO * db_po
    P_po = point_on_segment(B, T, Lb)
    D_closest_po, dd_po, _ = find_closest_defender(defenders, P_po)
    Theta_d_po = angle_deg(D_closest_po, P_po)
    Theta_b_po = angle_deg(B, P_po)

    # Return all metrics as a dictionary
    metrics = {
        'Theta_a_OBR': Theta_a,
        'Theta_d_OBR': Theta_d,
        'Theta_b_OBR': Theta_b,
        'da_OBR': da,
        'dd_OBR': dd,
        'db_OBR': db,
        'Theta_d_PO': Theta_d_po,
        'Theta_b_PO': Theta_b_po,
        'dd_PO': dd_po,
        'db_PO': db_po,
    }

    return metrics


import pandas as pd
import numpy as np
import requests
def load_match_data(match_id):
    """
    Load and process tracking and event data for a given match_id.

    Parameters:
    match_id (int): The match identifier

    Returns:
    tuple: Two pandas DataFrames (po_obr, tracking)
    """

    # ************
    # Load tracking data
    # ************
    tracking_data_github_url = f'https://media.githubusercontent.com/media/SkillCorner/opendata/741bdb798b0c1835057e3fa77244c1571a00e4aa/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl'
    raw_data = pd.read_json(tracking_data_github_url, lines=True)

    raw_df = pd.json_normalize(raw_data.to_dict('records'), 'player_data',
                               ['frame', 'timestamp', 'period', 'possession', 'ball_data'])

    # Extract 'player_id' and 'group from the 'possession' dictionary
    raw_df['possession_player_id'] = raw_df['possession'].apply(lambda x: x.get('player_id'))
    raw_df['possession_group'] = raw_df['possession'].apply(lambda x: x.get('group'))

    # (Optional) Expand the ball_data with json_normalize
    raw_df[['ball_x', 'ball_y', 'ball_z', 'is_detected_ball']] = pd.json_normalize(raw_df.ball_data)

    # (Optional) Drop the original 'possession' column if you no longer need it
    raw_df = raw_df.drop(columns=['possession', 'ball_data'])

    # Add the match_id identifier to the dataframe
    raw_df['match_id'] = match_id
    tracking_df = raw_df.copy()

    # Filter for frames with a player in possession
    tracking_df = tracking_df[tracking_df['possession_player_id'].notnull()]

    # ************
    # Load match data
    # ************

    def time_to_seconds(time_str):
        if time_str is None:
            return 90 * 60  # 120 minutes = 7200 seconds
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s

    meta_data_github_url = f'https://raw.githubusercontent.com/SkillCorner/opendata/741bdb798b0c1835057e3fa77244c1571a00e4aa/data/matches/{match_id}/{match_id}_match.json'
    # Read the JSON data as a JSON object
    response = requests.get(meta_data_github_url)
    raw_match_data = response.json()

    # The output has nested json elements. We process them
    raw_match_df = pd.json_normalize(raw_match_data, max_level=2)
    raw_match_df['home_team_side'] = raw_match_df['home_team_side'].astype(str)

    players_df = pd.json_normalize(
        raw_match_df.to_dict('records'),
        record_path='players',
        meta=['home_team_score', 'away_team_score', 'date_time',
              'home_team_side',
              'home_team.name', 'home_team.id',
              'away_team.name', 'away_team.id'
              ],
    )

    # Take only players who played and create their total time
    players_df = players_df[~((players_df.start_time.isna()) & (players_df.end_time.isna()))]
    players_df['total_time'] = players_df['end_time'].apply(time_to_seconds) - players_df['start_time'].apply(
        time_to_seconds)

    # Create a flag for GK
    players_df['is_gk'] = players_df['player_role.acronym'] == 'GK'

    # Add a flag if the given player is home or away
    players_df['match_name'] = players_df['home_team.name'] + ' vs ' + players_df['away_team.name']

    # Add a flag if the given player is home or away
    players_df['home_away_player'] = np.where(players_df.team_id == players_df['home_team.id'], 'Home', 'Away')

    # Create flag from player
    players_df['team_name'] = np.where(players_df.team_id == players_df['home_team.id'], players_df['home_team.name'],
                                       players_df['away_team.name'])

    # Figure out sides
    players_df[['home_team_side_1st_half', 'home_team_side_2nd_half']] = players_df['home_team_side'].astype(
        str).str.strip('[]').str.replace("'", '').str.split(', ', expand=True)

    # Clean up sides
    players_df['direction_player_1st_half'] = np.where(players_df.home_away_player == 'Home',
                                                       players_df.home_team_side_1st_half,
                                                       players_df.home_team_side_2nd_half)
    players_df['direction_player_2nd_half'] = np.where(players_df.home_away_player == 'Home',
                                                       players_df.home_team_side_2nd_half,
                                                       players_df.home_team_side_1st_half)

    # Clean up and keep the columns that we want to keep
    columns_to_keep = ['id', 'is_gk', 'direction_player_1st_half', 'direction_player_2nd_half', 'home_team.name',
                       'away_team.name', 'team_name', 'team_id']
    players_df = players_df[columns_to_keep]

    # ************
    # Merge tracking and match data
    # ************
    enriched_tracking_data = tracking_df.merge(players_df,
                                               left_on=['player_id'],
                                               right_on=['id'])

    # ************
    # Normalize attacking direction (right to left)
    # ************

    filtered_df = enriched_tracking_data.copy()

    # Direction for each row
    filtered_df['direction_player'] = np.where(
        filtered_df['period'] == 1,
        filtered_df['direction_player_1st_half'],
        filtered_df['direction_player_2nd_half']
    )

    # Mask for rows to flip (normalize to right_to_left)
    mask_flip = filtered_df['direction_player'] == 'right_to_left'

    # Flip coordinates for rows that don't have right_to_left direction
    filtered_df.loc[mask_flip, ['x', 'y', 'ball_x', 'ball_y']] *= -1

    # Possession info
    filtered_df['possession_team_name'] = np.where(
        filtered_df['possession_group'] == 'home team',
        filtered_df['home_team.name'],
        filtered_df['away_team.name']
    )
    filtered_df['possession_flag'] = np.where(
        filtered_df['possession_team_name'] == filtered_df['team_name'],
        'IP', 'OOP'
    )

    # Final selection
    columns_to_keep = [
        'match_id', 'x', 'y', 'player_id', 'frame', 'possession_player_id',
        'ball_x', 'ball_y', 'ball_z', 'is_gk', 'possession_flag', 'team_id'
    ]
    tracking_df = filtered_df[columns_to_keep].copy()

    # ************
    # Load Event data
    # ************

    url = (
        "https://raw.githubusercontent.com/SkillCorner/opendata/master/"f"data/matches/{match_id}/{match_id}_dynamic_events.csv")
    dynamic_events = pd.read_csv(url)
    # list(dynamic_events.columns)

    # Filter into separate DataFrames by event_type
    po = dynamic_events[dynamic_events['event_type'] == 'passing_option'].copy()
    obr = dynamic_events[dynamic_events['event_type'] == 'off_ball_run'].copy()

    # Filter obr and po dataframes
    columns_to_keep = ['match_id', 'event_id', 'player_id', 'frame_start', 'frame_end', 'xthreat']
    obr = obr[columns_to_keep]
    po = po[columns_to_keep]

    # ************
    # Align Tracking data and Event data Analysis
    # ************

    # Find start and end frames in obr and po that are in tracking_df
    tracking_frames = set(tracking_df['frame'])

    # Create a new column indicating if both frames match
    obr['both_frames_match'] = obr.apply(
        lambda row: row['frame_start'] in tracking_frames and row['frame_end'] in tracking_frames,
        axis=1
    )
    obr['event_type'] = 'off_ball_run'

    po['both_frames_match'] = po.apply(
        lambda row: row['frame_start'] in tracking_frames and row['frame_end'] in tracking_frames,
        axis=1
    )
    po['event_type'] = 'passing_option'

    # Combine obr and po events
    obr = obr[obr['both_frames_match'] == True]
    po = po[po['both_frames_match'] == True]
    po_obr = pd.concat([po, obr], ignore_index=True)
    po_obr = po_obr.drop('both_frames_match', axis=1)

    # Filter tracking_df for frames that are in po_obr
    target_frames = set(po_obr['frame_start']).union(set(po_obr['frame_end']))
    filtered_df = tracking_df[tracking_df['frame'].isin(target_frames)]
    tracking = filtered_df.copy()

    return po_obr, tracking


import math
from dataclasses import dataclass
from typing import List, Tuple
@dataclass
class Point:
    x: float
    y: float

def distance(p1: Point, p2: Point) -> float:
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def angle_deg(p_from: Point, p_to: Point) -> float:
    dx = p_to.x - p_from.x
    dy = p_to.y - p_from.y
    theta_rad = math.atan2(dy, dx)
    theta_deg = math.degrees(theta_rad)
    if theta_deg < 0:
        theta_deg += 360.0
    return theta_deg

def point_on_segment(p_start: Point, p_end: Point, dist_from_start: float) -> Point:
    seg_len = distance(p_start, p_end)
    if seg_len == 0:
        return Point(p_start.x, p_start.y)
    t = dist_from_start / seg_len
    return Point(
        p_start.x + t * (p_end.x - p_start.x),
        p_start.y + t * (p_end.y - p_start.y),
    )

def find_closest_defender(defenders: List[Point], P: Point) -> Tuple[Point, float, int]:
    """
    Returns:
        closest_defender: Point
        min_dist: float
        idx: index of closest defender in the original list
    """
    min_dist = float("inf")
    closest_def = None
    closest_idx = -1

    for i, D in enumerate(defenders):
        d = distance(D, P)
        if d < min_dist:
            min_dist = d
            closest_def = D
            closest_idx = i

    return closest_def, min_dist, closest_idx




