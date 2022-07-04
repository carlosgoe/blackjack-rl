import pandas as pd
import numpy as np


class Strategy:

    def __init__(self, hard=None, soft=None, src_hard=None, src_soft=None):
        # Get strategy table for player not having a usable ace
        if src_hard is not None:
            self.hard = pd.read_csv(src_hard, index_col=0).values
        elif hard is not None:
            self.hard = hard
        else:
            # Initialize table and set last row (player's sum is 21) to all 0s (=stick)
            self.hard = np.zeros((18, 10), dtype=np.int8)
            self.hard[:-1, :] = 0
        # Get strategy table for player having a usable ace
        if src_soft is not None:
            self.soft = pd.read_csv(src_soft, index_col=0).values
        elif soft is not None:
            self.soft = soft
        else:
            # Initialize table and set last row (player's sum is 21) to all 0s (=stick)
            self.soft = np.zeros((10, 10), dtype=np.int8)
            self.soft[:-1, :] = 0

    def action(self, players_sum, dealers_card, usable_ace):
        # Read action from strategy table
        if usable_ace:
            return self.soft[min(players_sum - 12, 9), dealers_card - 1]
        else:
            return self.hard[min(players_sum - 4, 17), dealers_card - 1]

    def match(self, strategy_compare):
        # Get strategy tables
        tables = np.concatenate([self.hard, self.soft])
        tables_compare = np.concatenate([strategy_compare.hard, strategy_compare.soft])
        # Iterate over indices
        sum_match = 0
        for i in range(tables.shape[0]):
            for j in range(tables.shape[1]):
                values = {tables[i, j], tables_compare[i, j]}
                # If primary and secondary action are equal, add 1 to comparisons
                if len(values) == 1:
                    sum_match += 1
                # If only one of them matches the other, add 0.5
                elif values == {2, 4} or values == {3, 6}:
                    sum_match += 0.5
                elif values == {0, 2} or values == {0, 3} or values == {1, 4} or values == {1, 6}:
                    sum_match += 0.5
                elif values == {2, 3} or values == {4, 6}:
                    sum_match += 0.5
        # Return the mean of comparisons
        return sum_match / tables.size

    def output(self, target_hard=None, target_soft=None):
        # Store string data in dictionaries
        data_hard, data_soft = {}, {}
        symbols = {0: 'S', 1: 'H', 2: 'DS', 3: 'RS', 4: 'DH', 6: 'RH'}
        for dealers_card in range(1, 10):
            column_name = 'A' if dealers_card == 1 else '{:02d}'.format(dealers_card)
            data_hard[column_name] = [symbols[a] for a in self.hard[:, dealers_card - 1]]
            data_soft[column_name] = [symbols[a] for a in self.soft[:, dealers_card - 1]]
        # Create pandas data frames from dictionaries
        df_hard = pd.DataFrame(data_hard, index=['{:02d}'.format(c) for c in range(4, 22)])
        df_soft = pd.DataFrame(data_soft, index=range(12, 22))
        # Print data frames
        print('Hard:\n{}\n\nSoft:\n{}'.format(df_hard.to_string(), df_soft.to_string()))
        # Export data frames to csv files
        if target_hard is not None:
            df_hard.to_csv(target_hard)
        if target_soft is not None:
            df_soft.to_csv(target_soft)
