import numpy as np
import pandas as pd
from typing import Union, Optional


def asi(
    high: pd.Series,
    low: pd.Series, 
    close: pd.Series,
    open_: pd.Series,
    limit_move_value: Union[float, str] = "auto",
    limit_move_pct: float = 0.10,
    offset: int = 0,
) -> pd.DataFrame:
    """
    Accumulative Swing Index (ASI) - Implementation TradingView precision
    
    L'ASI est développé par J. Welles Wilder pour isoler les "véritables" 
    mouvements de prix en comparant les relations entre les prix actuels 
    et ceux de la période précédente.
    
    Args:
        high (pd.Series): Série des plus hauts
        low (pd.Series): Série des plus bas
        close (pd.Series): Série des clôtures  
        open_ (pd.Series): Série des ouvertures
        limit_move_value (Union[float, str]): Valeur limite T.
            - "auto" : Calcul automatique selon le type d'actif
            - float : Valeur fixe
        limit_move_pct (float): Pourcentage pour calcul auto (défaut: 10%)
        offset (int): Décalage des résultats
        
    Returns:
        pd.DataFrame: DataFrame avec ASI et SI
        
    Raises:
        ValueError: Si les séries n'ont pas la même longueur
        
    Example:
        >>> df = pd.DataFrame({
        ...     'high': [105, 110, 108, 112, 115],
        ...     'low': [95, 100, 98, 102, 105], 
        ...     'close': [100, 108, 105, 110, 113],
        ...     'open': [98, 102, 107, 104, 111]
        ... })
        >>> result = asi(df['high'], df['low'], df['close'], df['open'])
        >>> print(result[['ASI', 'SI']])
    """
    
    # Validation des entrées
    if not (len(high) == len(low) == len(close) == len(open_)):
        raise ValueError("Toutes les séries doivent avoir la même longueur")
    
    # Conversion en float64 pour précision TradingView
    high = high.astype(np.float64)
    low = low.astype(np.float64)
    close = close.astype(np.float64)
    open_ = open_.astype(np.float64)
    
    # Calcul des valeurs précédentes (shift)
    high_prev = high.shift(1)
    low_prev = low.shift(1)
    close_prev = close.shift(1)
    open_prev = open_.shift(1)
    
    # Calcul de T (limit move value)
    if isinstance(limit_move_value, str) and limit_move_value == "auto":
        T = close_prev * limit_move_pct
    else:
        T = pd.Series(limit_move_value, index=close.index, dtype=np.float64)

    a = (high - close_prev).abs().astype(np.float64)
    b = (low - close_prev).abs().astype(np.float64)
    c = (high - low).abs().astype(np.float64)
    sh = (close_prev - open_prev).abs().astype(np.float64)

    K = pd.Series(np.maximum(a.to_numpy(), b.to_numpy()), index=close.index, dtype=np.float64)

    cond1 = (a >= b) & (a >= c)
    cond2 = (b >= a) & (b >= c)
    R = pd.Series(
        np.where(
            cond1.to_numpy(),
            (a - 0.5 * b + 0.25 * sh).to_numpy(),
            np.where(
                cond2.to_numpy(),
                (b - 0.5 * a + 0.25 * sh).to_numpy(),
                (c + 0.25 * sh).to_numpy(),
            ),
        ),
        index=close.index,
        dtype=np.float64,
    )

    numerator = (close_prev - close) + 0.5 * (close_prev - open_prev) + 0.25 * (close - open_)

    safe_R = R.where(R != 0.0)
    safe_T = T.where(T != 0.0)
    SI = 50.0 * (numerator / safe_R) * (K / safe_T)
    SI = SI.fillna(0.0)
    
    # Calcul de l'Accumulative Swing Index (ASI)
    ASI = SI.cumsum()
    
    # Création du DataFrame de résultats
    result = pd.DataFrame({
        'SI': SI,
        'ASI': ASI,
        'K': K,
        'T': T,
        'R': R
    }, index=close.index)
    
    # Application du décalage
    if offset != 0:
        result = result.shift(offset)
    
    return result


def asi_signals(
    df: pd.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low', 
    close_col: str = 'close',
    open_col: str = 'open',
    limit_move_value: Union[float, str] = "auto",
    limit_move_pct: float = 0.10,
) -> pd.DataFrame:
    """
    Générateur de signaux basés sur l'ASI
    
    Args:
        df (pd.DataFrame): DataFrame OHLC
        high_col (str): Nom colonne high
        low_col (str): Nom colonne low
        close_col (str): Nom colonne close
        open_col (str): Nom colonne open
        limit_move_value (Union[float, str]): Valeur limite T
        limit_move_pct (float): Pourcentage pour calcul auto
        
    Returns:
        pd.DataFrame: DataFrame avec ASI et signaux
    """
    
    # Calcul de l'ASI
    asi_result = asi(
        df[high_col],
        df[low_col],
        df[close_col], 
        df[open_col],
        limit_move_value,
        limit_move_pct
    )
    
    # Ajout au DataFrame original
    result = df.copy()
    result['ASI'] = asi_result['ASI']
    result['SI'] = asi_result['SI']
    
    # Calcul des signaux
    result['ASI_trend'] = np.where(result['ASI'] > result['ASI'].shift(1), 1, -1)
    result['ASI_positive'] = result['ASI'] > 0
    result['ASI_negative'] = result['ASI'] < 0
    
    # Signaux de base
    result['buy_signal'] = (
        (result['ASI_trend'] == 1) & 
        (result['ASI_positive']) & 
        (result['ASI_trend'].shift(1) == -1)
    )
    
    result['sell_signal'] = (
        (result['ASI_trend'] == -1) & 
        (result['ASI_negative']) & 
        (result['ASI_trend'].shift(1) == 1)
    )
    
    # Signaux de breakout (fracture de zeros)
    result['bull_breakout'] = (
        (result['ASI'] > 0) & 
        (result['ASI'].shift(1) <= 0)
    )
    
    result['bear_breakout'] = (
        (result['ASI'] < 0) & 
        (result['ASI'].shift(1) >= 0)
    )
    
    return result


# Configuration pour différents marchés
MARKET_CONFIGS = {
    'crypto': {
        'limit_move_pct': 0.10,  # 10% pour cryptos
        'description': 'Cryptocurrency markets'
    },
    'forex': {
        'limit_move_pct': 0.02,   # 2% pour forex
        'description': 'Forex markets'
    },
    'stocks': {
        'limit_move_pct': 0.05,   # 5% pour actions
        'description': 'Stock markets'
    },
    'futures': {
        'limit_move_pct': 0.07,   # 7% pour futures (S&P500)
        'description': 'Futures markets'
    }
}


def asi_by_market(
    df: pd.DataFrame,
    market: str = 'crypto',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close', 
    open_col: str = 'open',
) -> pd.DataFrame:
    """
    Calcul ASI avec configuration par marché
    
    Args:
        df (pd.DataFrame): DataFrame OHLC
        market (str): Type de marché ('crypto', 'forex', 'stocks', 'futures')
        high_col (str): Nom colonne high
        low_col (str): Nom colonne low
        close_col (str): Nom colonne close
        open_col (str): Nom colonne open
        
    Returns:
        pd.DataFrame: DataFrame avec ASI et signaux
    """
    
    if market not in MARKET_CONFIGS:
        raise ValueError(f"Market '{market}' not supported. Use: {list(MARKET_CONFIGS.keys())}")
    
    config = MARKET_CONFIGS[market]
    
    return asi_signals(
        df=df,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        open_col=open_col,
        limit_move_value="auto",
        limit_move_pct=config['limit_move_pct']
    )


if __name__ == "__main__":
    # Exemple d'utilisation
    data = {
        'open': [100.0, 102.0, 107.0, 104.0, 111.0, 109.0],
        'high': [105.0, 110.0, 108.0, 112.0, 115.0, 113.0], 
        'low': [95.0, 100.0, 98.0, 102.0, 105.0, 103.0],
        'close': [100.0, 108.0, 105.0, 110.0, 113.0, 108.0]
    }
    
    df = pd.DataFrame(data)
    
    # Test avec marché crypto
    result = asi_by_market(df, market='crypto')
    print("Résultats ASI pour marché crypto:")
    print(result[['ASI', 'SI', 'ASI_trend', 'buy_signal', 'sell_signal']].round(4))
