import random
from copy import copy
import numpy as np

# Define necessary functions/constants
has_usable_ace = lambda hand: 1 in hand and sum(hand) <= 11
get_sum = lambda hand: sum(hand) + 10 if has_usable_ace(hand) else sum(hand)
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


class Blackjack:

    def __init__(self, seed=None):
        # All attributes are None because the game hasn't been started
        self.done, self.dealer_visible, self.reward, self.first_round = [None] * 4
        self.player, self.dealer, self.players_sum, self.dealers_sum, self.usable_ace = [None] * 5
        # Use seed to load same game multiple times
        random.seed(seed)

    def reset(self, players_sum=None, dealers_card=None, usable_ace=None):
        # Game is not over, dealer's second card is invisible, reward is 0, and it's the first round
        self.done = False
        self.dealer_visible = False
        self.reward = 0
        self.first_round = True
        # Iterate randomly over all card combinations and return the one that fulfills the criteria
        first_card, second_card = copy(deck), copy(deck)
        random.shuffle(first_card)
        random.shuffle(second_card)
        for card1 in first_card:
            for card2 in second_card:
                self.player = [card1, card2]
                self.players_sum = get_sum(self.player)
                self.usable_ace = has_usable_ace(self.player)
                sum_criteria = self.players_sum < 21 and (players_sum is None or self.players_sum == players_sum)
                usable_ace_criteria = usable_ace is None or self.usable_ace == usable_ace
                if sum_criteria and usable_ace_criteria:
                    break
            else:
                continue
            break
        # Give first card to dealer
        self.dealer = []
        if dealers_card is not None and 1 <= dealers_card <= 10:
            self.dealer.append(dealers_card)
        else:
            self.dealer.append(random.choice(deck))
        # Give second card to dealer and make sure there is no Blackjack
        if self.dealer[0] == 1:
            self.dealer.append(random.choice(deck[:9]))
        elif self.dealer[0] == 10:
            self.dealer.append(random.choice(deck[1:]))
        else:
            self.dealer.append(random.choice(deck))
        self.dealers_sum = get_sum(self.dealer)
        # Return initial observation and invalid actions
        return self.__observation(), self.__invalid_actions()

    def step(self, action):
        # Quit if game is already over or action is invalid
        if self.done or not self.first_round and (action == 2 or action == 3):
            return
        # Player draws card if action is 1 (=hit) or 2 (=double)
        if action == 1 or action == 2:
            self.player.append(random.choice(deck))
            self.players_sum = get_sum(self.player)
            self.usable_ace = has_usable_ace(self.player)
        # Dealer draws cards if action is 2 (=double) or 0 (=stick) and player didn't bust
        if (action == 2 or action == 0) and self.players_sum <= 21:
            while self.dealers_sum < 17:
                self.dealer.append(random.choice(deck))
                self.dealers_sum = get_sum(self.dealer)
            self.dealer_visible = True
        # Update game over and first round status
        self.done = self.players_sum > 21 or action != 1
        self.first_round = False
        # Reward of -0.5 if action is 3 (=surrender)
        if action == 3:
            self.reward = -0.5
        # Reward of -1 if player busts or sum_dealer > sum_player
        elif self.done and (self.players_sum > 21 or self.players_sum < self.dealers_sum <= 21):
            self.reward = -1
        # Reward of 1 if dealer busts or sum_player <= 21 > sum_dealer
        elif self.done and (self.dealers_sum > 21 or self.dealers_sum < self.players_sum <= 21):
            self.reward = 1
        # Reward of 0 if game is not over or sum_player == sum_dealer <= 21
        else:
            self.reward = 0
        # If player doubled, multiply reward by two
        if action == 2:
            self.reward *= 2
        # Return observation, reward, done, invalid actions
        return self.__observation(), self.reward, self.done, self.__invalid_actions()

    def show(self):
        # Print dealer's card(s and sum)
        if self.dealer_visible:
            print('Dealer:', ', '.join([str(c) for c in self.dealer]))
            print('->', self.dealers_sum)
        elif self.dealer:
            print('Dealer: %i, ?' % self.dealer[0])
        # Print player's cards and sum
        if self.player:
            print('\nPlayer:', ', '.join([str(c) for c in self.player]))
            # In case of usable ace, print both sums
            if self.usable_ace and not self.done:
                print('-> {}/{}\n'.format(sum(self.player), self.players_sum))
            else:
                print('-> {}\n'.format(self.players_sum))
        # If game is over, print winner depending on sign of last reward
        if self.done:
            winner = int(np.sign(self.reward))
            print('Winner: {}\n'.format(['None', 'Player', 'Dealer'][winner]))

    def __observation(self):
        # Create one-hot encoded input for player's sum
        arr_player = np.zeros(18, dtype=np.uint8)
        arr_player[min(self.players_sum - 4, 17)] = 1
        # Create one-hot encoded input for dealer's visible card
        arr_dealer = np.zeros(10, dtype=np.uint8)
        arr_dealer[self.dealer[0] - 1] = 1
        # Create array indicating if there is a usable ace and if it's the first round
        flags = np.array([self.usable_ace, self.first_round], dtype=np.uint8)
        # Return all arrays as one concatenation
        return np.concatenate([arr_player, arr_dealer, flags])

    def __invalid_actions(self):
        # Make 3 (=surrender) and 2 (=double) invalid if it is not the first round
        invalid_actions = set()
        if not self.first_round:
            invalid_actions |= {2, 3}
        # Make 1 (=hit), 2 (=double), and 3 (=surrender) invalid if card sum is 21 and 0 (=stick) if it is less than 12
        if self.players_sum == 21:
            invalid_actions |= {1, 2, 3}
        elif self.players_sum <= 11:
            invalid_actions.add(0)
        return list(invalid_actions)
