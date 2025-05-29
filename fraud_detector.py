import pandas as pd
import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model("fraud.json")


def make_prediction(
    step=1,
    amount=10000.0,
    oldbalanceOrg=20000.0,
    newbalanceOrig=10000.0,
    oldbalanceDest=0.0,
    newbalanceDest=10000.0,
    balanceDiffOrig=10000.0,
    balanceDiffDest=10000.0,
    type_CASH_OUT=1,
    type_TRANSFER=0,
):
    data = pd.DataFrame(
        [
            {
                "step": step,
                "amount": amount,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "balanceDiffOrig": balanceDiffOrig,
                "balanceDiffDest": balanceDiffDest,
                "type_CASH_OUT": type_CASH_OUT,
                "type_TRANSFER": type_TRANSFER,
            }
        ]
    )

    prediction = model.predict(data)[0]
    # Predict
    if prediction == 0:
        return False
    else:
        return True


print(make_prediction())
