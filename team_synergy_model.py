import numpy as np

class TeamSynergySpatialModel:
    """
    A basic implementation of a spatial factor model for measuring team synergy,
    inspired by "Uncovering the sources of team synergy" (Brave et al., 2019).
    
    This model attempts to quantify the indirect effects of teammate interactions
    (synergy) on overall team performance beyond the sum of individual contributions (e.g. WAR).
    """
    def __init__(self, rho=0.5):
        """
        Initialize the spatial factor model.
        :param rho: Spatial autoregressive parameter representing the strength of 
                    network interactions (spillover effects) among teammates.
        """
        self.rho = rho
        
    def fit_team_synergy(self, player_war: np.ndarray, adjacency_matrix: np.ndarray):
        """
        Calculate team synergy using a spatial autoregressive (SAR) approach.
        
        y = ρWy + Xβ + ε
        In a simplified factor model, the synergy factor (spillover) can be estimated by
        evaluating the spatial multiplier: (I - ρW)^-1
        
        :param player_war: 1D array of individual player metrics (e.g., WAR) for a team.
        :param adjacency_matrix: 2D array (W) representing the strength of interaction 
                                 between players (e.g., playing time together, positional links).
                                 Usually row-normalized.
        :return: A tuple containing the total team value, the baseline sum, and the synergy component.
        """
        n_players = len(player_war)
        identity = np.eye(n_players)
        
        # Calculate the spatial multiplier matrix: (I - ρW)^(-1)
        try:
            spatial_multiplier = np.linalg.inv(identity - self.rho * adjacency_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("The matrix (I - ρW) is singular. Check the value of rho and W.")
            
        # The synergistic player contributions considering the network
        synergistic_war = spatial_multiplier @ player_war
        
        baseline_team_value = np.sum(player_war)
        total_team_value = np.sum(synergistic_war)
        synergy_component = total_team_value - baseline_team_value
        
        return total_team_value, baseline_team_value, synergy_component, synergistic_war

def generate_sample_data(n_players=9):
    """
    Generate synthetic baseball data for 9 players to test the synergy model.
    """
    np.random.seed(42)
    player_war = np.random.uniform(0.5, 5.0, n_players)
    
    W = np.random.uniform(0, 1, (n_players, n_players))
    np.fill_diagonal(W, 0)
    
    # Row normalize
    row_sums = W.sum(axis=1)
    W_normalized = W / row_sums[:, np.newaxis]
    
    return player_war, W_normalized
