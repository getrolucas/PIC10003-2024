import numpy as np
import pandas as pd
from typing import Union


class Evaluation:
    """
    Classe para avaliação de modelos de regressão, fornecendo as principais métricas de desempenho:
    - MAE: Mean Absolute Error (Erro Absoluto Médio)
    - MSE: Mean Squared Error (Erro Quadrático Médio)
    - RMSE: Root Mean Squared Error (Raiz do Erro Quadrático Médio)
    - R²: Coeficiente de Determinação (R-squared)
    
    Construção das métricas baseada no texto: *Métricas para Regressão: Entendendo as métricas R², MAE, MAPE, MSE e RMSE.*
    Disponível em: https://medium.com/data-hackers/prevendo-n%C3%BAmeros-entendendo-m%C3%A9tricas-de-regress%C3%A3o-35545e011e70
    
    Parameters
    ---
    y : Union[pd.Series, np.array]
        Valores reais observados.
    y_pred: Union[pd.Series, np.array]
        Valores preditos pelo modelo.
    
    Methods
    ---
    summary():
        Retorna um DataFrame com todas as métricas de avaliação.
    """
    
    def __init__(self, 
                 y : Union[pd.Series, np.ndarray], 
                 y_pred : Union[pd.Series, np.ndarray]) -> None:
        
        self.y = np.array(y)
        self.y_pred = np.array(y_pred)
        self.y_bar = np.mean(self.y)
        
        
    def _mae(self) -> np.float64:
        """
        Erro absoluto médio (Mean Absolute Error [MAE])
        """
        return np.mean(np.abs(self.y - self.y_pred))

    
    def _mse(self) -> np.float64:
        """
        Erro quadrático médio (Mean Squared Error [MSE])
        """
        return np.mean((self.y - self.y_pred) ** 2)
    
    
    def _rmse(self) -> np.float64:
        """
        Raiz do Erro Quadrático Médio (Root Mean Squared Error [RMSE])
        """
        return np.sqrt(self._mse())
    
    
    def _r2(self) -> np.float64:
        """
        Coeficiente de determinação (Coefficient of Determination [R²])
        """
        dnm = np.sum((self.y - self.y_bar) ** 2)
        nm = np.sum((self.y - self.y_pred) ** 2)
        return 1 - np.divide(dnm, nm)

    
    def summary(self) -> pd.DataFrame:
        """
        Returns
        ---
        pd.DataFrame com sumário de desempenho do modelo.
        """
        metrics = {
            "MAE": self._mae(),
            "MSE": self._mse(),
            "RMSE": self._rmse(),
            "R2": self._r2()
        }
        return pd.DataFrame(metrics, index=[0])