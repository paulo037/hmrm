USERS_CHECKIN_FILENAME_BASELINE_HMRM="/media/gegen07/Expansion/data/mestrado/checkins_data/checkins_local/checkins_local_Alabama.csv"
FEATURES_FILENAME_HMRM="out/features.csv"

BASELINES_CONFIG='{
          "job": "hmrm_baseline",
          "users_checkin_filename": "'$USERS_CHECKIN_FILENAME_BASELINE_HMRM'",
          "features_filename": "'$FEATURES_FILENAME_HMRM'",
          }'

python3 main.py "${BASELINES_CONFIG}"