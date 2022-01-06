from random import randint
from BoardClasses import Move
from BoardClasses import Board
import math
import random
# from copy import copy
from copy import deepcopy

# The following part should be completed by students.
# Students can modify anything except the class name and exisiting functions and varibles.

'''
turn = 1 Black
turn = 2 White
UCT = (wi + h(n)) / ln(c * sp/si))

compute_score rules:
1. check number of checkers black/white & king(+10)/man(+3)
2. Space Matters (-0.5)/(+0.5)/(+1.5)
3. Capture checkers (+6)
4. Move Position (abs(target.row - start.row))
'''

depth = 99

class GameTree():

    def __init__(self, turn, move=None, parent=None, board=None):
        self.hvalue = 0 #T_n
        self.board = board  # board class
        self.parent = parent
        # self.root = root
        self.unvisited = list()
        self.children = list()  # key is the "move", value is "node"
        self.color = turn
        self.oppo = {1: 2, 2: 1}
        self.move = move
        self.wi = 0  # wi/_ # times lead to its parent wins
        self.si = 0  # _/si # Visited value
        self.ucb = 0


    def reach_leaf(self, moves_count):
        return len(self.children) == moves_count


class StudentAI():

    def __init__(self, col, row, p):
        self.col = col
        self.row = row
        self.p = p
        self.board = Board(col, row, p)
        self.board.initialize_game()
        self.color = ''
        self.opponent = {1: 2, 2: 1}
        self.color = 2

    def get_move(self, move):  # move is oppo move
        #  AI.get_move((0,1)-(1,0)), what move can my AI get after knowing the opponent move
        if len(move) != 0:
            self.board.make_move(move, self.opponent[self.color])  # opponent choose its move
            # self.board.show_board() # the board after oppo moves
        else:
            self.color = 1

        moves = self.board.get_all_possible_moves(self.color)  # the move AI can take


        index = None
        inner_index = None

        root = GameTree(turn=self.color, move=move, board=self.board)
        top_root_color = root.color
        best_move_node = self.MCTS(root, depth)
        for i in range(len(moves)):
            for j in range(len(moves[i])):
                if moves[i][j].seq == best_move_node.move.seq:
                    index = i
                    inner_index = j

        move = moves[index][inner_index]
        '''
        index = randint(0,len(moves)-1) # modify
        inner_index =  randint(0,len(moves[index])-1)
        move = moves[index][inner_index]
        '''
        self.board.make_move(move, self.color)

        return move

    def random_next_board(self, node):  # node = pickchild in 1st iteration

        moves = node.board.get_all_possible_moves(node.color)

        state_hvalue = {}
        for move in moves:
            for m in move:
                node.board.make_move(m, node.color)
                cur_board_h = self.compute_heuristic(node)
                cur_move_h = self.compute_move_heuristic(node, m)
                d_key = cur_board_h + cur_move_h
                if d_key not in state_hvalue:
                    state_hvalue[d_key] = [m]
                    node.board.undo()
                else:
                    state_hvalue[d_key].append(m)
                    node.board.undo()

        if len(set(state_hvalue.keys())) == 1:
            pick_move = random.choice(list(state_hvalue.values())[0])
        else:
            if node.color == self.color:
                max_hvalue = max(state_hvalue.keys())
                pick_move = random.choice(state_hvalue[max_hvalue])
            else:
                min_hvalue = min(state_hvalue.keys())
                pick_move = random.choice(state_hvalue[min_hvalue])
        node.board.make_move(pick_move, node.color)

        return node


    def select(self, node):
        if node.color == 1:
            color = "B"
            oppocolor = "W"
        else:
            color = "W"
            oppocolor = "B"

        if node.board.is_win(oppocolor) not in [-1, 1, 2]:
            if node.children == []: # which means we reach the leaf node, so EXPAND
                children = []
                curr_moves = node.board.get_all_possible_moves(node.color)
                for moves in curr_moves:
                    for m in moves:
                        node.board.make_move(m, node.color) # board after root make move
                        copyboard = deepcopy(node.board)
                        node.board.undo()
                        children.append(GameTree(self.opponent[node.color], m, node, copyboard))

                node.unvisited.extend(children)

            largest_ucb = float("-inf")
            if node.unvisited != []:
                pick = node.unvisited[0]
                node.children.append(pick)
                node.unvisited.remove(pick)

            else:
                for i in node.children:
                    if i.si == 0:
                        i.ucb = float("inf")
                    else:
                        i.ucb = (i.wi + i.hvalue) / i.si + 1.414*math.sqrt(math.log(i.parent.si)/i.si)
                    if i.ucb > largest_ucb:
                        largest_ucb = i.ucb  # infinity
                        pick = i  # A

            if pick.si == 0: # pick is NOT full_expand,  do rollout

                if pick.color == 1:
                    color = "B"
                    oppocolor = "W"
                else:
                    color = "W"
                    oppocolor = "B"

                if pick.board.is_win(oppocolor) in [-1, 1, 2]:
                    winner = pick.board.is_win(oppocolor)
                    deterministic_node_color = pick.color
                    winner_board_heuristic = self.compute_heuristic(pick)
                    move_count = 0
                    self.backprop(pick, deterministic_node_color, winner_board_heuristic, winner, move_count)
                else:
                    pickchild = self.expand(pick)
                    winner_board_heuristic, winner, deterministic_node_color, move_count= self.roll_out(pickchild) # the last node
                    # do backprop
                    self.backprop(pickchild, deterministic_node_color, winner_board_heuristic, winner, move_count)
                    pickchild.children = []

            else: # pick is full_expand, finished roll_out
                self.select(pick)

        else:

            deterministic_node_color = node.color
            winner_board_heuristic = self.compute_heuristic(node)
            winner = node.board.is_win(oppocolor)
            self.backprop(node, deterministic_node_color, winner_board_heuristic, winner, 0)

        return



    def expand(self, picknode): # node = pick

        moves_for_picknode = picknode.board.get_all_possible_moves(picknode.color)

        index, inner_index = None, None
        if len(moves_for_picknode) > 1:
            index = randint(0, len(moves_for_picknode) - 1)  # modify
        else:
            index = 0

        if len(moves_for_picknode[index]) > 1:
            inner_index = randint(0, len(moves_for_picknode[index]) - 1)
        else:
            inner_index = 0

        action = moves_for_picknode[index][inner_index]
        pickchild_board = deepcopy(picknode.board) # first initialize
        pickchild = GameTree(self.opponent[picknode.color], action, picknode, pickchild_board)
        picknode.children.append(pickchild)
        pickchild.board.make_move(action, picknode.color)  # board after root make move
        return pickchild


    def best_selection(self, node):

        best_ucb = node.children[0].ucb
        best_child = node.children[0]

        for child_node in node.children:
            if child_node.ucb > best_ucb:
                best_ucb = child_node.ucb
                best_child = child_node

        return best_child


    def MCTS(self, root, depth):

        for i in range(depth):
            self.select(root)

        best_next_move = self.best_selection(root)

        return best_next_move


    def backprop(self, pick_child, deter_node_color, heuristic, winner, move_count):  # backprop(expand_node)
        # winner == color
        num = 0
        while num < move_count:
            pick_child.board.undo()
            # pick_child.color = self.opponent[pick_child.color]
            num += 1

        while pick_child != None:
            # increment the si: visited times
            pick_child.si += 1

            # add heuristic score
            pick_child.hvalue += heuristic

            if winner == -1:
                if pick_child.color != self.color:
                    pick_child.wi += 1

            else:
                if self.color == winner:
                    if pick_child.color != winner:
                        pick_child.wi += 1

                if self.color != winner: # top_node_color = B, winner = W
                    if pick_child.color == winner: # pick_child.color = w
                        pass
                    else: # pick_child.color != winner: # pick_child.color = B
                        pick_child.wi += 1

            pick_child = pick_child.parent


    def roll_out(self, node):  # Roll_out(pick_child color = 1)
        winner = None

        if node.color==1:
            oppocolor = "W"
        else:
            oppocolor = "B"
        count_rollout = 0  # count_rollout = how many time we make move on pick_child ==> undo
        while node.board.is_win(oppocolor) not in [-1, 1, 2]:
            next_node = self.random_next_board(node) # new_node = original node + new_board(after move)
            node = next_node
            count_rollout += 1

            node.color = self.opponent[node.color]
            if node.color == 1:
                oppocolor = "W"
            else:
                oppocolor = "B"

        winner_board_heuristic = self.compute_heuristic(node)

        if node.board.is_win(oppocolor) == 1: # node = deterministic node
            winner = 1
        if node.board.is_win(oppocolor) == 2:
            winner = 2
        if node.board.is_win(oppocolor) == -1:
            winner = -1 # tie

        return winner_board_heuristic, winner, node.color, count_rollout # node.color = deterministic node turn/color

    def compute_move_heuristic(self, node, move):
        score = 0
        if abs(move[-1][0] - move[0][0]) > 1:  # [(6,5),(4,3),(2,1)]
            capture_num = len(move.seq) - 1
            score += capture_num * 3
        target_point_row = move[-1][0]
        target_point_col = move[-1][1]
        original_point_color = node.board.board[move[0][0]][
            move[0][1]].color.lower()
        if original_point_color == "w":
            original_point_color = 2
        else:
            original_point_color = 1

        if self.opponent[original_point_color] == "2":
            opponent_color = "W"
        else:
            opponent_color = "B"

        # print("board row and col: ", self.board.row, self.board.col)
        # print("target row and col: ", target_point_row, target_point_col)

        if target_point_row + 1 < self.board.row and target_point_col + 1 < self.board.col:  # which direction
            if node.board.board[target_point_row + 1][target_point_col + 1].color == opponent_color:  # right down
                if target_point_row - 1 >= 0 and target_point_col - 1 >= 0:
                    if node.board.board[target_point_row - 1][target_point_col - 1].color == ".":  # left up
                        score -= 5
        if target_point_row + 1 < self.board.row and target_point_col - 1 >= 0:
            if node.board.board[target_point_row + 1][target_point_col - 1].color == opponent_color:  # left down
                if target_point_row - 1 >= 0 and target_point_col + 1 < self.board.col:
                    if node.board.board[target_point_row - 1][target_point_col + 1].color == ".":  # right up
                        score -= 5
        if target_point_row - 1 >= 0 and target_point_col - 1 >= 0:
            if node.board.board[target_point_row - 1][target_point_col - 1].color == opponent_color:  # left up
                if target_point_row + 1 < self.board.row and target_point_col + 1 < self.board.col:
                    if node.board.board[target_point_row + 1][target_point_col + 1].color == ".":  # right down
                        score -= 5
        if target_point_row - 1 >= 0 and target_point_col + 1 < self.board.col:
            if node.board.board[target_point_row - 1][target_point_col + 1].color == opponent_color:  # right up
                if target_point_row + 1 < self.board.row and target_point_col - 1 >= 0:
                    if node.board.board[target_point_row + 1][target_point_col - 1].color == ".":  # left down
                        score -= 5

                # self.board[target_point_row+1][target_point_col+1]   # right down
                # self.board[target_point_row+1][target_point_col-1]   # left down
                # self.board[target_point_row-1][target_point_col-1]   # left up
                # self.board[target_point_row-1][target_point_col+1]   # right up
        return score

    def compute_heuristic(self, node):  # Heuristic Score(Final State Win/Lose Board)
        board_score = 0
        # if node.color==self.color:
        #     board_score += 15
        # else:
        #     board_score -= 15

        # black +
        # White -
        black_king, white_king = set(), set()  # set([1, 1], [2, 4])
        black_man, white_man = set(), set()

        for i in range(node.board.row):
            for j in range(node.board.col):
                if node.board.board[i][j].color == "B":
                    if node.board.board[i][j].is_king:
                        black_king.add((i, j))
                    else:
                        black_man.add((i, j))
                if node.board.board[i][j].color == "W":
                    if node.board.board[i][j].is_king:
                        white_king.add((i, j))
                    else:
                        white_man.add((i, j))

        # space matters
        if self.col % 2 == 1:  # when self.col is odd, has precise middle point
            middle_point = math.floor(self.col / 2)  # (-1.5)
            quartile = math.floor(middle_point / 2)
            # left and right centered (-0.5)
            deduct_point = [i for i in range(quartile)]
            deduct_point += [i for i in range(self.col - quartile, self.col)]
            # left most and right most sided (+1.5)
            add_most = [i for i in range(quartile, middle_point)]
            add_most += [i for i in range(middle_point + 1, self.col - quartile)]
        else:
            middle_point = int(self.col / 2)  # (-1.5)
            # left and right centered (-0.5)
            quartile = math.floor((middle_point - 1) / 2)
            deduct_point = [i for i in range(quartile)]
            deduct_point += [i for i in range(self.col - quartile, self.col)]
            # left most and right most sided (+1.5)
            add_most = [i for i in range(quartile, middle_point - 1)]
            add_most += [i for i in range(middle_point + 1, self.col - quartile)]

        # count black/white king num
        for bk in black_king:
            board_score += 0.5  # 10
            # space matter
            if bk[1] in add_most:
                board_score += 1.5
            elif bk[1] in deduct_point:
                board_score -= 0.5
            else:
                board_score -= 1.5

        for wk in white_king:
            board_score -= 0.5 # 10
            # space matter
            if wk[1] in add_most:
                board_score -= 1.5
            elif wk[1] in deduct_point:
                board_score += 0.5
            else:
                board_score += 1.5

        # count capture points
        if abs(node.move[-1][0] - node.move[0][0]) > 1: #[(1,1)-(1,5), (2,2)-(3,4)]
            capture_num = len(node.move.seq) - 1
            if node.color == 1:
                board_score += capture_num * 3
            if node.color == 2:
                board_score -= capture_num * 3

        # count opponent move points
        if node.color == 1:
            board_score += (abs(node.move.seq[-1][0] - node.move.seq[0][0]))
        if node.color == 2:
            board_score -= (abs(node.move.seq[-1][0] - node.move.seq[0][0]))

        # count black/white man num
        for bm in black_man:
            board_score += 0.2
            # space matter
            if bm[1] in add_most:
                board_score += 0.5
            elif bm[1] in deduct_point:
                board_score -= 0.2
            else:
                board_score -= 0.5
        for wm in white_man:
            board_score -= 0.2
            if wm[1] in add_most:
                board_score -= 0.5
            elif wm[1] in deduct_point:
                board_score += 0.2
            else:
                board_score += 0.5

        if self.color == 2:
            board_score *= -1

        return board_score