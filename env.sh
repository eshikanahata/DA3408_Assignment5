export PYTHONPATH="$PWD/partA:$PWD/partB:$PWD/partC:$PWD/partD:$PWD"
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export MLFLOW_DISABLE_AGENT_HINT=1
echo "PYTHONPATH and MLFLOW_TRACKING_URI set for $PWD"
