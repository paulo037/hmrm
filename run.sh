POI_DETECTION_FILENAME="/home/guilherme/Documentos/users_detected_pois.csv"
GROUND_TRUTH="/home/guilherme/Documentos/Users_filtrados_por_regiao/Users_rotulated/rotulated_manual_edit.csv"
USERS_CHECKIN_FILENAME_BASELINE_HMRM="data/checkins_local/checkins_local_Alabama.csv"
FEATURES_FILENAME_HMRM="out/features.csv"

USERS_STEPS_FILENAME="/home/guilherme/Documentos/Users_filtrados_por_regiao/frequent_users.csv"
POI_CLASSIFICATION_FILENAME="/home/guilherme/Documentos/users_classified_pois.csv"

BASELINES_CONFIG='{
          "job": "hmrm_baseline",
          "poi_detection_filename": "'$POI_DETECTION_FILENAME'",
          "users_steps_filename": "'$USERS_STEPS_FILENAME_BASELINE_HMRM'",
          "ground_truth": "'$GROUND_TRUTH'",
          "users_checkin_filename": "'$USERS_CHECKIN_FILENAME_BASELINE_HMRM'",
          "features_filename": "'$FEATURES_FILENAME_HMRM'",
          "weeplaces?": "No"
          }'

python3 main.py "${BASELINES_CONFIG}"