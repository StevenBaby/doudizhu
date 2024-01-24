import os
import numpy as np

from colorama import (
    init as colorama_init,
    Fore,
    Style,
    Back,
)

from douzero.env.game import GameEnv
from douzero.env.game import InfoSet
from douzero.env.game import bombs
from douzero.evaluation.deep_agent import DeepAgent
from douzero.evaluation import simulation as sim

from logger import logger

dirname = os.path.dirname(os.path.abspath(__file__))


models = {
    'landlord': os.path.join(dirname, "baselines/douzero_WP/landlord.ckpt"),
    'landlord_up': os.path.join(dirname, "baselines/douzero_WP/landlord_up.ckpt"),
    'landlord_down': os.path.join(dirname, "baselines/douzero_WP/landlord_down.ckpt")
}

EnvCard2RealCard = {3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
                    8: '8', 9: '9', 10: 'T', 11: 'J', 12: 'Q',
                    13: 'K', 14: 'A', 17: '2', 20: 'X', 30: 'D'}

RealCard2EnvCard = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                    '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12,
                    'K': 13, 'A': 14, '2': 17, 'X': 20, 'D': 30}

AllEnvCard = [i for i in range(3, 15) for _ in range(4)]
AllEnvCard.extend([17 for _ in range(4)])
AllEnvCard.extend([20, 30])


def render_action(action):
    cards = [EnvCard2RealCard[a] for a in action]
    return "".join(cards)


def paste_action(string):
    action = []
    if string in {'n', 'pass'}:
        return action
    for ch in string:
        if ch not in RealCard2EnvCard:
            return None
        action.append(RealCard2EnvCard[ch])
    return action


def generate_data():
    deck = AllEnvCard.copy()
    np.random.shuffle(deck)
    card_play_data = {'landlord': deck[:20],
                      'landlord_up': deck[20:37],
                      'landlord_down': deck[37:54],
                      'three_landlord_cards': deck[17:20],
                      }
    for key in card_play_data:
        card_play_data[key].sort()
    return card_play_data


class MyEnv(GameEnv):

    def step(self, action=None):
        if action is None:
            action = self.players[self.acting_player_position].act(
                self.game_infoset)

        if len(action) > 0:
            self.last_pid = self.acting_player_position

        if action in bombs:
            self.bomb_num += 1

        self.last_move_dict[
            self.acting_player_position] = action.copy()

        self.card_play_action_seq.append(action)
        self.update_acting_player_hand_cards(action)

        self.played_cards[self.acting_player_position] += action

        if self.acting_player_position == 'landlord' and \
                len(action) > 0 and \
                len(self.three_landlord_cards) > 0:
            for card in action:
                if len(self.three_landlord_cards) > 0:
                    if card in self.three_landlord_cards:
                        self.three_landlord_cards.remove(card)
                else:
                    break

        self.game_done()
        if not self.game_over:
            self.get_acting_player_position()
            self.game_infoset = self.get_infoset()

        return action


def input_action(info_sets):
    if len(info_sets['landlord'].legal_actions) == 1:
        return info_sets['landlord'].legal_actions[0]
    while True:
        info = f"{Fore.GREEN}{len(info_sets['landlord_up'].player_hand_cards)} " \
        f"{Fore.RED}{render_action(info_sets['landlord'].player_hand_cards)} " \
        f"{Fore.MAGENTA}{len(info_sets['landlord_down'].player_hand_cards)} {Style.RESET_ALL}"
        print(info)
        action = paste_action(input("INPUT: "))
        if action is None:
            continue
        if not action:
            return action
        for a in info_sets['landlord'].legal_actions:
            if tuple(a) == tuple(action):
                return action
        continue


def play():
    # 输入玩家的牌
    data = generate_data()

    logger.info("loading models....")
    players = sim.load_card_play_models(models)

    env = MyEnv(players)
    env.card_play_init(data)
    idx = 1

    colors = [Fore.GREEN, Fore.RED, Fore.MAGENTA]

    while not env.game_over:
        action = None
        if idx % 3 == 1:
            action = input_action(env.info_sets)

        action = env.step(action)
        action = render_action(action)
        if not action:
            action = 'pass'
        print(f"{colors[idx % 3]}{action}")
        idx += 1

    info = f"{Fore.GREEN}{render_action(env.info_sets['landlord_up'].player_hand_cards)} " \
        f"{Fore.RED}{render_action(env.info_sets['landlord'].player_hand_cards)} " \
        f"{Fore.MAGENTA}{render_action(env.info_sets['landlord_down'].player_hand_cards)} {Style.RESET_ALL}"
    print(info)

    env.reset()

    print(env.num_wins['landlord'],
          env.num_wins['farmer'],
          env.num_scores['landlord'],
          env.num_scores['farmer'])


def main():
    colorama_init()
    play()


if __name__ == '__main__':
    main()
