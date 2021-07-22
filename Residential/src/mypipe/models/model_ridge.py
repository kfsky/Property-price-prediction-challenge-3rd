from sklearn.linear_model import Ridge
from .base import BaseModel


class MyRidgeModel(BaseModel):
    def build_model(self):
        model = Ridge(**self.params)
        return model
