import numpy as np
import pandas as pd
from scipy.optimize import minimize

class BraveEtAlSynergyModel:
    """
    Implementation of the Spatial Factor Model for MLB Team Synergy
    based on "Uncovering the sources of team synergy" (Brave et al., 2019).
    
    This model computes player-specific productivity residuals based on their
    playing time and models them using a spatial autoregressive (SAR) structure 
    to capture teammate interactions.
    """
    
    def __init__(self, alpha=50.0):
        """
        :param alpha: Expected wins for a team of replacement level players (~50 wins).
        """
        self.alpha = alpha
        self.rho = None
        
    def expected_win_contribution(self, team_wins, kappa, is_pitcher):
        """
        Calculates a player's expected contribution to team wins based on playing time.
        Formula (7): \hat{W}_{int} = \eta_{it} \tau_{it} (W_{nt} - \hat{\alpha})
        
        :param team_wins: Total actual wins of the team.
        :param kappa: Appearance weights for each player (\tau proxy).
        :param is_pitcher: Boolean array indicating if the player is a pitcher.
        """
        tau = np.zeros_like(kappa)
        hitters = ~is_pitcher
        pitchers = is_pitcher
        
        # Normalize appearance weights separately for hitters and pitchers
        if np.sum(hitters) > 0:
            tau[hitters] = kappa[hitters] / np.sum(kappa[hitters])
        if np.sum(pitchers) > 0:
            tau[pitchers] = kappa[pitchers] / np.sum(kappa[pitchers])
            
        # \eta ratio for apportioning league wins to pitchers vs hitters (0.43 vs 0.57)
        eta = np.where(is_pitcher, 0.43, 0.57) 
        
        # Expected wins
        w_hat = eta * tau * (team_wins - self.alpha)
        return w_hat
        
    def construct_adjacency_matrix(self, kappa):
        """
        Constructs the adjacency matrix A based on appearance weights \kappa.
        A_{ijt} = \kappa_{it} + \kappa_{jt} for i \neq j, 0 otherwise.
        """
        n = len(kappa)
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    A[i, j] = kappa[i] + kappa[j]
                    
        # Row-normalize the matrix to create proper spatial weights
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0 
        A_norm = A / row_sums
        return A_norm

    def fit(self, player_war, kappa, is_pitcher, team_wins):
        """
        Fits the spatial model to calculate synergy metrics.
        
        Process:
        1. Calculate Expected Wins (\hat{W})
        2. Calculate Productivity Residuals (\hat{\epsilon} = \hat{W} - WAR)
        3. Construct Adjacency Matrix A
        4. Estimate spatial correlation \rho by modeling \hat{\epsilon} = \rho A \hat{\epsilon} + v
        """
        n = len(player_war)
        
        # 1. Expected Wins
        w_hat = self.expected_win_contribution(team_wins, kappa, is_pitcher)
        
        # 2. Player Productivity Residuals (Formula 4)
        residuals = w_hat - player_war
        
        # 3. Teammate Interaction Network
        A = self.construct_adjacency_matrix(kappa)
        
        # 4. Estimate \rho via Sum of Squared Errors of fundamental shocks (v)
        # v = (I - \rho A) \hat{\epsilon}
        def objective(rho):
            I = np.eye(n)
            v = (I - rho * A) @ residuals
            return np.sum(v**2)
            
        res = minimize(objective, x0=0.1, bounds=[(-0.99, 0.99)])
        self.rho = res.x[0]
        
        # Calculate components
        I = np.eye(n)
        v = (I - self.rho * A) @ residuals
        
        # Synergy effect (spillover) = \rho A \hat{\epsilon} = \hat{\epsilon} - v
        # This maps to pcWAR (Player Complementarity WAR) - the net impact on teammates
        pcWAR = residuals - v 
        
        # Total Team Synergy (tcWAR)
        tcWAR = np.sum(pcWAR)
        
        return {
            'expected_wins': w_hat,
            'productivity_residuals': residuals,
            'fundamental_shocks': v,
            'pcWAR': pcWAR,
            'tcWAR': tcWAR,
            'rho': self.rho,
            'A': A
        }

def generate_sample_data(n_players=15):
    """
    Generate synthetic baseball data for a team of 15 players (10 hitters, 5 pitchers)
    """
    np.random.seed(42)
    is_pitcher = np.array([False]*10 + [True]*5)
    
    # Base WAR
    player_war = np.random.uniform(-0.5, 5.0, n_players)
    
    # Kappa (appearance weights - proxy for PA and Innings Pitched)
    kappa = np.random.uniform(0.1, 1.0, n_players)
    kappa[is_pitcher] = kappa[is_pitcher] * 0.5 
    
    # Actual Team Wins (simulating a slight synergy over-performance)
    team_wins = int(np.sum(player_war) + 50 + np.random.normal(2, 4))
    
    return player_war, kappa, is_pitcher, team_wins
