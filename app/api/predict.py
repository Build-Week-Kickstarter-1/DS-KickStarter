import logging
import random

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator, confloat

log = logging.getLogger(__name__)
router = APIRouter()


class Campaign(BaseModel):
    """Use this data model to parse the request body JSON."""

    name: str
    category: str
    currency: str
    deadline: str
    goal: confloat(ge=0)
    launched: str

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""

        df = pd.DataFrame([dict(self)])
        df['deadline'] = pd.to_datetime(df['deadline'])
        df['launched'] = pd.to_datetime(df['launched'])
        return df



@router.post('/predict')
async def predict(campaign: Campaign):
    """
    Make random baseline predictions for classification problem 🔮

    ### Request Body
    - `name`: The name of the Kickstarter campaign
    - `category`: The main category of the Kickstarter campaign
    - `currency`: The three letter abbreviation of the currency the Kickstarter is based in
    - `deadline`: The end date of the Kickstarter campaign
    - `goal`: The funding goal of the Kickstarter campaign
    - `launched`: The start date of the Kickstarter campaign

    ### Response
    - `prediction`: boolean, at random
    - `predict_proba`: float between 0.5 and 1.0, 
    representing the predicted class's probability

    Replace the placeholder docstring and fake predictions with your own model.
    """

    X_new = campaign.to_df()
    log.info(X_new)
    y_pred = None
    if y_pred == True:
        return 'The campaign should succeed!'
    elif y_pred == False:
        return 'The campaign will probably fail.'