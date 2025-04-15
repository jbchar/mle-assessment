from pydantic import BaseModel
from typing import Optional


"""
"Bonus: the basic model only uses a subset of the columns provided in the
house sales data.
Create an additional API endpoint where only the required features have
to be provided in order to get a prediction."

This pydantic model meets the above requirement without the need 
for an additional endpoint
"""

class InputFeatures(BaseModel):
    zipcode: str
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int
    #the rest of these features are Optional because they are not used in the model
    # but are included in the data for completeness and to allow for forward compatibility
    id: Optional[int] = None
    date: Optional[str] = None
    # including Optional price to allow for comparison of real value with predicted value
    # this is not a feature used in the model, but can be used for evaluation
    price: Optional[float] = None
    waterfront: Optional[int] = None
    view: Optional[int] = None
    condition: Optional[int] = None
    grade: Optional[int] = None
    yr_built: Optional[int]
    yr_renovated: Optional[int] = None
    lat: Optional[float]
    long: Optional[float]
    sqft_living15: Optional[int]
    sqft_lot15: Optional[int]
    # Add new features here