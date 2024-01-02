import tkinter as tk
import math
import random
import time
import numpy as np

class Puissance4GUI:
    def __init__(self, root, rows=6, cols=7, display=False):
        self.rows = rows
        self.cols = cols
        self.root = root
        self.cell_size = 60
        self.board = [[' ' for _ in range(cols)] for _ in range(rows)]
        self.current_player = 'X'
        self.user_turn = True
        self.branch = 0
        self.value = -1

        if display:
            self.canvas = tk.Canvas(self.root, width=cols * self.cell_size, height=rows * self.cell_size)
            self.canvas.pack()
            self.canvas.bind("<Button-1>", self.on_click)
            self.draw_board()

    def change_current_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def is_game_over(self):
        return self.is_winner('X') or self.is_winner('O') or self.is_draw()

    def on_click(self, event):
        if not self.user_turn:
            return

        col = event.x // self.cell_size
        if self.is_valid_move(col):
            self.make_move(col)
            self.draw_board()

            if self.is_winner(self.current_player):
                print(f"Player {self.current_player} wins!")
                self.canvas.unbind("<Button-1>")
            elif self.is_draw():
                print("It's a draw!")
                self.canvas.unbind("<Button-1>")
            else:
                self.change_current_player()
                self.user_turn = False

    def is_valid_move(self, col):
        return 0 <= col < self.cols and self.board[0][col] == ' '

    def make_move(self, col):
        if not self.is_valid_move(col):
            return False
        for row in range(self.rows - 1, -1, -1):
            if self.board[row][col] == ' ':
                self.board[row][col] = self.current_player
                break
        self.change_current_player()
        return True

    def is_winner(self, player):
        # Check horizontally
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row][col + i] == player for i in range(4)):
                    return True

        # Check vertically
        for col in range(self.cols):
            for row in range(self.rows - 3):
                if all(self.board[row + i][col] == player for i in range(4)):
                    return True

        # Check diagonally (bottom-left to top-right)
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row - i][col + i] == player for i in range(4)):
                    return True

        # Check diagonally (top-left to bottom-right)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(self.board[row + i][col + i] == player for i in range(4)):
                    return True

        return False

    def is_draw(self):
        return all(self.board[0][col] != ' ' for col in range(self.cols))

    def draw_board(self):
        self.canvas.delete("all")
        for row in range(self.rows):
            for col in range(self.cols):
                self.draw_cell(row, col, self.board[row][col])

    def draw_cell(self, row, col, symbol):
        x0, y0 = col * self.cell_size, row * self.cell_size
        x1, y1 = x0 + self.cell_size, y0 + self.cell_size
        self.canvas.create_rectangle(x0, y0, x1, y1, fill="grey")

        if symbol == 'X':
            self.canvas.create_oval(x0 + 10, y0 + 10, x1 - 10, y1 - 10, fill="red")
        elif symbol == 'O':
            self.canvas.create_oval(x0 + 10, y0 + 10, x1 - 10, y1 - 10, fill="yellow")

    def get_available_moves(self):
        return [col for col in range(self.cols) if self.is_valid_move(col)]

    def copy(self):
        copy_game = Puissance4GUI(self.root, self.rows, self.cols, False)
        copy_game.board = [row[:] for row in self.board]
        copy_game.current_player = self.current_player
        copy_game.value = self.value
        copy_game.branch = self.branch
        return copy_game

    def print_board(self):
        # clear_output(wait=True)  # Efface le contenu précédent
        max_token_width = max(len(str(i)) for i in range(10))
        for row in self.board:
            print('|'.join(cell.center(max_token_width) if cell != ' ' else ' ' * max_token_width for cell in row))
            print('-' * (self.cols * (max_token_width + 1) - 1))
        print(' '.join(str(i).center(max_token_width) for i in range(self.cols)))

class Agent:

    generation = 0

    def __init__(self, game, symbol = 'O'):
        self.game = game
        self.symbol = symbol
        self.nb_symb_max = self.game.cols * self.game.rows
        self.neurones = Neurone()
        self.matrice = self.matrice_gaussienne(game.rows, game.cols, 2)
        self.poids_A = [1, 1, 1, 1] # 0 <-> 1
        self.poids_B_1 = [0, 0, 0, 0] # -1 <-> 1
        self.poids_B_2 = [0, 0, 0, 0] # -1 <-> 1
        self.victory = 0
        self.defeat = 0

    def copy(self):
        copy_game = Agent(self.game, self.symbol)
        copy_game.matrice = self.matrice
        copy_game.poids_A = self.poids_A
        copy_game.poids_B_1 = self.poids_B_1
        copy_game.poids_B_2 = self.poids_B_2
        copy_game.victory = self.victory
        copy_game.defeat = self.defeat
        return copy_game

    def matrice_gaussienne(self, taille_x, taille_y, ecart_type):
        # Crée une matrice de la taille spécifiée remplie de zéros
        matrice = np.zeros((taille_x, taille_y))

        # Calcule les indices du centre de la matrice
        centre_x, centre_y = taille_x / 2 - 0.5, taille_y / 2 - 0.5


        # Applique une distribution gaussienne bidimensionnelle à la matrice
        for i in range(taille_x):
            for j in range(taille_y):
                distance_x = i - centre_x
                distance_y = j - centre_y
                matrice[i, j] = np.exp(-0.5 * ((distance_x / ecart_type)**2 + (distance_y / ecart_type)**2))

        # Normalise la matrice pour que la valeur centrale soit 1
        matrice[math.ceil(centre_x), math.ceil(centre_y)] = 1.0
        matrice[math.floor(centre_x), math.ceil(centre_y)] = 1.0
        matrice[math.ceil(centre_x), math.floor(centre_y)] = 1.0
        matrice[math.floor(centre_x), math.floor(centre_y)] = 1.0

        return matrice

    def minimax(self, depth, maximizing_player, symbol, game, alpha, beta):
        if depth == 0 or game.is_game_over():
            if symbol == 'O' : notsymb = 'X'
            else : notsymb = 'O'
            if game.is_winner(symbol):
                return math.inf
            elif game.is_winner(notsymb):
                return -math.inf
            else:
                return self.evaluate_game(game, symbol)

        if maximizing_player:
            max_eval = -math.inf
            for move in game.get_available_moves():
                temp_game = game.copy()
                temp_game.make_move(move)
                eval = self.minimax(depth - 1, False, symbol, temp_game, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in game.get_available_moves():
                temp_game = game.copy()
                temp_game.make_move(move)
                eval = self.minimax(depth - 1, True, symbol, temp_game, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def play_best_move_V0(self, depth):
        best_move = None
        best_eval = -float(math.inf)
        best_game = []
        for move in self.game.get_available_moves():
            temp_game = self.game.copy()
            temp_game.make_move(move)
            eval = self.minimax(depth-1, False, self.symbol, temp_game, -math.inf, math.inf)
            if eval > best_eval or move==0:
                best_eval = eval
                best_move = move
        
        if best_eval < 0 and depth > 1:
            self.play_best_move_V0(depth - 1)
        elif best_move is not None:
            self.game.make_move(best_move)
        else:
            for i in range(self.game.cols):
                if self.game.is_valid_move(i):
                    self.game.make_move(i)
                    break

    def play_best_move_V1(self, depth, nb_move_kept):
        best_game = []
        av_move = 0
        for move in range(self.game.cols):
            if self.game.is_valid_move(move):
                av_move += 1
                temp_game = self.game.copy()
                temp_game.branch = move
                temp_game.make_move(move)
                temp_game.value = self.minimax(0, False, self.symbol, temp_game, -math.inf, math.inf)  # peut etre -inf et inf le pb
                best_game.append(temp_game)


        best_game = sorted(best_game, key=lambda game: game.value, reverse=True)

        if (av_move - nb_move_kept) > 0:
            for i in range(av_move - nb_move_kept):
                best_game.pop()

        print("before")
        for i in range(len(best_game)):
            print("best_game branch : ", best_game[i].branch, "with the value : ", best_game[i].value)
        print("")

        best_move = None
        best_eval = -math.inf
        for move in range(len(best_game)):
            temp_game = best_game[move].copy()
            eval = self.minimax(depth - 1, False, self.symbol, temp_game, -math.inf, math.inf)
            best_game[move].value = eval
            if eval > best_eval or move==0:
                best_eval = eval
                best_move = best_game[move].branch
        best_game = sorted(best_game, key=lambda game: game.value, reverse=True)

        print("after")
        for i in range(len(best_game)):
            print("best_game branch : ", best_game[i].branch, "with the value : ", best_game[i].value)
        print("")

        if best_eval < 0 and depth > 1:
            self.play_best_move_V1(depth - 1, nb_move_kept)
        elif best_move is not None:
            self.game.make_move(best_move)
        else:
            for i in range(self.game.cols):
                if self.game.is_valid_move(i):
                    self.game.make_move(i)
                    break
             
    def play_best_move_test(self, depth, nb_move_kept):
        best_game = []
        av_move = 0
        for move in range(self.game.cols):
            if self.game.is_valid_move(move):
                av_move += 1
                temp_game = self.game.copy()
                temp_game.branch = move
                temp_game.make_move(move)
                temp_game.value = self.minimax(0, False, self.symbol, temp_game, -math.inf, math.inf)  # peut etre -inf et inf le pb
                best_game.append(temp_game)


        best_game = sorted(best_game, key=lambda game: game.value, reverse=True)

        if (av_move - nb_move_kept) > 0:
            for i in range(av_move - nb_move_kept):
                best_game.pop()

        print("before")
        for i in range(len(best_game)):
            print("best_game branch : ", best_game[i].branch, "with the value : ", best_game[i].value)
        print("")

        best_move = None
        best_eval = -math.inf
        for move in range(len(best_game)):
            temp_game = best_game[move].copy()
            #temp_game.make_move(move)
            #temp_game.change_current_player()
            eval = self.minimax(depth - 1, False, self.symbol, temp_game, -math.inf, math.inf)
            best_game[move].value = eval
            if eval > best_eval or move==0:
                best_eval = eval
                best_move = best_game[move].branch
        best_game = sorted(best_game, key=lambda game: game.value, reverse=True)

        print("after")
        for i in range(len(best_game)):
            print("best_game branch : ", best_game[i].branch, "with the value : ", best_game[i].value)
        print("")

        if best_eval < 0 and depth > 1:
            self.play_best_move_V1(depth - 1, nb_move_kept)
        elif best_move is not None:
            self.game.make_move(best_move)
        else:
            for i in range(self.game.cols):
                if self.game.is_valid_move(i):
                    self.game.make_move(i)
                    break

    def evaluate_game(self, game, symbol):
        if game.value==math.inf or game.value==-math.inf:
            return game.value

        total_score = 0.0
        nb_symb = self.neurones.number_of_symbol(game, 'X') + self.neurones.number_of_symbol(game, 'O')
        fact_symb = nb_symb / self.nb_symb_max

        total_score += (self.poids_A[0] * fact_symb + self.poids_B_1[0])* self.neurones.eval_neurone_1(game, symbol) + self.poids_B_2[0]
        total_score += (self.poids_A[1] * fact_symb + self.poids_B_1[1])* self.neurones.eval_neurone_2(game, symbol) + self.poids_B_2[1]
        total_score += (self.poids_A[2] * fact_symb + self.poids_B_1[2])* self.neurones.eval_neurone_3(game, symbol) + self.poids_B_2[2]
        total_score += (self.poids_A[3] * fact_symb + self.poids_B_1[3])* self.neurones.eval_neurone_4(game, symbol, self.matrice) + self.poids_B_2[3]

        return total_score

    @staticmethod
    def IA_against_IA(ag1, ag2, depth, nb_rand_move = 2):
        root = tk.Tk()
        game = Puissance4GUI(root, 6, 7, True)
        game = Agent.random_game(game, nb_rand_move)
        ag1.game = game
        ag2.game = game

        game.current_player = 'X'
        i=random.randint(0, 1)
        if i==0:
            ag1.symbol = 'X'
            ag2.symbol = 'O'
        else:
            ag1.symbol = 'O'
            ag2.symbol = 'X'

        while not(game.is_winner("O") or game.is_winner("X") or game.is_draw()):
            if i%2==0:
                ag1.play_best_move_V0(depth)
            else:
                ag2.play_best_move_V0(depth)
            i+=1

        if game.is_winner("O"):
            if ag1.symbol == "O":
                ag1.victory += 1
                ag2.defeat += 1
            else:
                ag2.victory += 1
                ag1.defeat += 1
        elif game.is_winner("X"):
            if ag1.symbol == "X":
                ag1.victory += 1
                ag2.defeat += 1
            else:
                ag2.victory += 1
                ag1.defeat += 1

    def IA_against_IA_V2(ag1, ag2, depth, nb_rand_move = 2):
        root = tk.Tk()
        game = Puissance4GUI(root, 6, 7, True)
        game = Agent.random_game(game, nb_rand_move)
        game_2 = game.copy()
        ag1.game = game
        ag2.game = game

        game.current_player = 'X'
        game_2.current_player = 'X'

        ag1.symbol = 'X'
        ag2.symbol = 'O'
        i=0
        while not(game.is_winner("O") or game.is_winner("X") or game.is_draw()):
            if i%2==0:
                ag1.play_best_move_V0(depth)
            else:
                ag2.play_best_move_V0(depth)
            i+=1

        ag1.game = game_2
        ag2.game = game_2
        ag1.symbol = 'O'
        ag2.symbol = 'X'

        i=1
        while not(game_2.is_winner("O") or game_2.is_winner("X") or game_2.is_draw()):
            if i%2==0:
                ag1.play_best_move_V0(depth)
            else:
                ag2.play_best_move_V0(depth)
            i+=1

        if game.is_winner("O"):
            if game_2.is_winner("X") or game_2.is_draw():
                ag2.victory += 1
                ag1.defeat += 1
        elif game.is_draw():
            if game_2.is_winner("O"):
                ag1.victory += 1
                ag2.defeat += 1
            elif game_2.is_winner("X"):
                ag2.victory += 1
                ag1.defeat += 1   
        else:
            if game_2.is_winner("O") or game_2.is_draw():
                ag1.victory += 1
                ag2.defeat += 1



    @staticmethod
    def match_of_IA(ag1, ag2, best_of, depth):
        ag1.victory = 0
        ag2.victory = 0
        for i in range(best_of):
            print("ag1.victory : ", ag1.victory, "ag2.victory : ", ag2.victory)
            print("")
            #Agent.IA_against_IA(ag1, ag2, depth)
            Agent.IA_against_IA_V2(ag1, ag2, depth)
            if ag1.victory > best_of // 2:
                return ag1 
            elif ag2.victory > best_of // 2:
                return ag2
            
        if ag1.victory > ag2.victory:
            return ag1
        else:
            return ag2

    @staticmethod # marche que pour les depth paire
    def random_game(game, depth):
        for i in range(depth):
            x = random.randint(0, 6)
            if game.is_valid_move(x):
                game.make_move(x)
        return game

    @staticmethod
    def training(best_of = 5, nb_loses = 7, depth = 2, learning_rate = 0.2, nb_of_bot = 100): # nb_loses : nombre de partie perdu avant d'etre eliminer
        while(True):
            agent_tab = []
            root = tk.Tk()
            game = Puissance4GUI(root, 6, 7, True)
            ag1 = Agent(game)
            ag2 = Agent(game)
            ag3 = Agent(game)
            ag1.load()
            ag2.load("poids_1.txt")
            ag3.load("poids_2.txt")
            agent_tab.append(ag1)
            agent_tab.append(ag2)
            agent_tab.append(ag3)

            if Agent.generation > 0:
                #mixe génétique entre ag1 et ag2
                for n in range(nb_of_bot//7):
                    agent = Agent(game)
                    agent.mix_gen(ag1, ag2)
                    agent_tab.append(agent.copy())

                #mixe génétique entre ag1 et ag3
                for n in range(nb_of_bot//7):
                    agent = Agent(game)
                    agent.mix_gen(ag1, ag3)
                    agent_tab.append(agent.copy())

                #mixe génétique entre ag2 et ag3
                for n in range(nb_of_bot//7):
                    agent = Agent(game)
                    agent.mix_gen(ag2, ag3)
                    agent_tab.append(agent.copy())
                
                #variante de ag1
                for i in range(nb_of_bot//7):
                    agent = ag1.copy()
                    for j in range(12):
                        rand = random.randint(1,10)
                        if rand < 7:
                            if random.randint(0,1):
                                agent.gen_change(j, Agent.amount(learning_rate), True)
                            else:
                                agent.gen_change(j, Agent.amount(learning_rate), False)
                        elif rand < 9:
                            if j < 4:
                                agent.change_gen(j, random.uniform(0, 1))
                            else:
                                agent.change_gen(j, random.uniform(-1, 1))  

                #variante de ag2
                for i in range(nb_of_bot//7):
                    agent = ag2.copy()
                    for j in range(12):
                        rand = random.randint(1,10)
                        if rand < 7:
                            if random.randint(0,1):
                                agent.gen_change(j, Agent.amount(learning_rate), True)
                            else:
                                agent.gen_change(j, Agent.amount(learning_rate), False)
                        elif rand < 9:
                            if j < 4:
                                agent.change_gen(j, random.uniform(0, 1))
                            else:
                                agent.change_gen(j, random.uniform(-1, 1))  
            
                #variante de ag3
                for i in range(nb_of_bot//7):
                    agent = ag3.copy()
                    for j in range(12):
                        rand = random.randint(1,10)
                        if rand < 7:
                            if random.randint(0,1):
                                agent.gen_change(j, Agent.amount(learning_rate), True)
                            else:
                                agent.gen_change(j, Agent.amount(learning_rate), False)
                        elif rand < 9:
                            if j < 4:
                                agent.change_gen(j, random.uniform(0, 1))
                            else:
                                agent.change_gen(j, random.uniform(-1, 1))   

            while(len(agent_tab) < nb_of_bot):
                agent = Agent(game)
                agent.random_poids()
                agent_tab.append(agent)

            for i in range(nb_of_bot):
                agent_tab[i].Print_poids()
                print("")
            
            while(len(agent_tab) > 3):
                print("avancement : ", (nb_of_bot-len(agent_tab))/(nb_of_bot-3)*100, "%")
                random.shuffle(agent_tab)
                for i in range(len(agent_tab)-1):
                    Agent.match_of_IA(agent_tab[i], agent_tab[i+1], Agent.Best_of(best_of), depth)
                    print("     ", (i/(len(agent_tab)-1))*100, "%")
                tab_pop = []
                tab_pop_exist = False
                for i in range(len(agent_tab)):
                    if agent_tab[i].defeat > Agent.Nb_loss(nb_loses):
                        tab_pop_exist = True
                        tab_pop.append(i)
                if tab_pop_exist:        
                    for i in range(len(tab_pop)):    
                        if 3 < len(agent_tab):
                            agent_tab.pop(tab_pop[i]-i)
                if(len(agent_tab)%2 == 1 and len(agent_tab) != 3):
                    agent = Agent(game)
                    agent.random_poids()
                    agent.defeat = Agent.Nb_loss(nb_loses)
                    agent_tab.append(agent)
                    
            agent_tab.sort(key=lambda agent: agent.defeat)
            
            Agent.generation += 1
            agent_tab[0].save("poids.txt")
            agent_tab[1].save("poids_1.txt")
            agent_tab[2].save("poids_2.txt")
            print("")
            print("generation : ", Agent.generation)
            print("")

    def get_gen(self, neurone):
        if neurone < 4:
            return self.poids_A[neurone]
        elif neurone < 8:
            return self.poids_B_1[neurone-4]
        else:
            return self.poids_B_2[neurone-8]
        
    def change_gen(self, neurone, new_gen):
        if neurone < 4:
            self.poids_A[neurone] = new_gen
        elif neurone < 8:
            self.poids_B_1[neurone-4] = new_gen
        else:
            self.poids_B_2[neurone-8] = new_gen
        
    @staticmethod
    def Best_of(best_of):
        x = 1 + 2*(Agent.generation//10)
        if x > best_of:
            return best_of
        else:
            return x
        
    @staticmethod
    def Nb_loss(nb_loss):
        x = 2*(Agent.generation//10)
        if x > nb_loss:
            return nb_loss
        else:
            return x

    def random_poids(self):
        self.poids_A = [random.uniform(0, 1) for i in range(4)]
        self.poids_B_1 = [random.uniform(-1, 1) for i in range(4)]
        self.poids_B_2 = [random.uniform(-1, 1) for i in range(4)]

    @staticmethod
    def amount(x = 0.2):
        x = x / (Agent.generation//10 + 1)
        return x

    def gen_change(self, neurone, amount, add = True):
        if neurone < 4:
            if add:
                if self.poids_A[neurone]+amount < 1:
                    self.poids_A[neurone] += amount
            else:
                if self.poids_A[neurone]-amount > 0:
                    self.poids_A[neurone] -= amount
        elif neurone < 8:
            if add:
                if self.poids_B_1[neurone-4]+amount < 1:
                    self.poids_B_1[neurone-4] += amount
            else:
                if self.poids_B_1[neurone-4]-amount > -1:
                    self.poids_B_1[neurone-4] -= amount
        else:
            if add:
                if self.poids_B_2[neurone-8]+amount < 1:
                    self.poids_B_2[neurone-8] += amount
            else:
                if self.poids_B_2[neurone-8]-amount > -1:
                    self.poids_B_2[neurone-8] -= amount

    def mix_gen(self, ag1, ag2):
        for i in range(4):
            if random.randint(0,1):
                self.poids_A[i] = ag1.poids_A[i]
            else:
                self.poids_A[i] = ag2.poids_A[i]
        for i in range(4):
            if random.randint(0,1):
                self.poids_B_1[i] = ag1.poids_B_1[i]
            else:
                self.poids_B_1[i] = ag2.poids_B_1[i]
        for i in range(4):
            if random.randint(0,1):
                self.poids_B_2[i] = ag1.poids_B_2[i]
            else:
                self.poids_B_2[i] = ag2.poids_B_2[i]

    def save(self, nom_fichier="poids.txt"):
        with open(nom_fichier, "w") as f:
            f.write(str(self.poids_A[0]) + " " + str(self.poids_A[1]) + " " + str(self.poids_A[2]) + " " + str(self.poids_A[3]) + "\n")
            f.write(str(self.poids_B_1[0]) + " " + str(self.poids_B_1[1]) + " " + str(self.poids_B_1[2]) + " " + str(self.poids_B_1[3]) + "\n")
            f.write(str(self.poids_B_2[0]) + " " + str(self.poids_B_2[1]) + " " + str(self.poids_B_2[2]) + " " + str(self.poids_B_2[3]) + "\n")

        with open("gen.txt", "w") as f:
            f.write(str(self.__class__.generation))
            
    def load(self, nom_fichier="poids.txt"):
        with open(nom_fichier, "r") as f:
            poids_line = f.readline()

            poids_str_list = poids_line.split()

            self.poids_A[0] = float(poids_str_list[0])
            self.poids_A[1] = float(poids_str_list[1])
            self.poids_A[2] = float(poids_str_list[2])
            self.poids_A[3] = float(poids_str_list[3])

            poids_line = f.readline()

            poids_str_list = poids_line.split()

            self.poids_B_1[0] = float(poids_str_list[0])
            self.poids_B_1[1] = float(poids_str_list[1])
            self.poids_B_1[2] = float(poids_str_list[2])
            self.poids_B_1[3] = float(poids_str_list[3])

            poids_line = f.readline()

            poids_str_list = poids_line.split()

            self.poids_B_2[0] = float(poids_str_list[0])
            self.poids_B_2[1] = float(poids_str_list[1])
            self.poids_B_2[2] = float(poids_str_list[2])
            self.poids_B_2[3] = float(poids_str_list[3])

            with open("gen.txt", "r") as f:
                self.__class__.generation = int(f.read())
            
    def Print_poids(self):
        print(self.poids_A)
        print(self.poids_B_1)
        print(self.poids_B_2)

class Neurone:

    @staticmethod
    def number_of_symbol(game, symbol):
        nb_symbol = 0
        for row in range(game.rows):
            for col in range(game.cols):
                if game.board[row][col] == symbol:
                    nb_symbol += 1
        return nb_symbol
    @staticmethod
    def eval_neurone_1(game, symbol):
        count = 0
        nb_symbol = 0
        for row in range(game.rows):
            for col in range(game.cols):
                if game.board[row][col] == symbol:
                    nb_symbol += 1
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue  # Ignorer la cellule actuelle
                            r, c = row + dr, col + dc
                            if 0 <= r < game.rows and 0 <= c < game.cols and game.board[r][c] == symbol:
                                count += 1
        if nb_symbol == 0: return 0
        return (count / (nb_symbol*4.45))

    @staticmethod
    def eval_neurone_2(game, symbol):
        count = 0
        nb_symbol = 0
        for row in range(game.rows):
            for col in range(game.cols):
                if game.board[row][col] == symbol:
                    nb_symbol += 1
                    # Variables pour le blocage dans chaque direction
                    blocked_horizontal = False
                    blocked_vertical = False
                    blocked_diagonal_right = False
                    blocked_diagonal_left = False
                    # Vérifier les chaînages en horizontal (gauche et droite)
                    horizontal_count = 0
                    for i in range(-3, 1):  # De gauche à droite
                        if 0 <= col + i < game.cols:
                            if game.board[row][col + i] == symbol:
                                horizontal_count += 1
                            elif game.board[row][col + i] != ' ':
                                # Bloqué par un symbole opposé, réinitialiser le compteur
                                blocked_horizontal = True
                                horizontal_count = 0

                    # Vérifier les chaînages en vertical (haut et bas)
                    vertical_count = 0
                    for i in range(-3, 1):  # De haut en bas
                        if 0 <= row + i < game.rows:
                            if game.board[row + i][col] == symbol:
                                vertical_count += 1
                            elif game.board[row + i][col] != ' ':
                                # Bloqué par un symbole opposé, réinitialiser le compteur
                                blocked_vertical = True
                                vertical_count = 0

                    # Vérifier les chaînages en diagonale (bas gauche vers haut droite)
                    diagonal_right_count = 0
                    for i in range(-3, 1):  # De bas gauche à haut droite
                        if 0 <= col + i < game.cols and 0 <= row - i < game.rows:
                            if game.board[row - i][col + i] == symbol:
                                diagonal_right_count += 1
                            elif game.board[row - i][col + i] != ' ':
                                # Bloqué par un symbole opposé, réinitialiser le compteur
                                blocked_diagonal_right = True
                                diagonal_right_count = 0

                    # Vérifier les chaînages en diagonale (haut gauche vers bas droite)
                    diagonal_left_count = 0
                    for i in range(-3, 1):  # De haut gauche à bas droite
                        if 0 <= col + i < game.cols and 0 <= row + i < game.rows:
                            if game.board[row + i][col + i] == symbol:
                                diagonal_left_count += 1
                            elif game.board[row + i][col + i] != ' ':
                                # Bloqué par un symbole opposé, réinitialiser le compteur
                                blocked_diagonal_left = True
                                diagonal_left_count = 0

                    # Si l'une des directions a un chaînage de 3 symboles non bloqués, augmenter le compteur
                    if not blocked_horizontal and horizontal_count == 3:
                        count += 1
                    if not blocked_vertical and vertical_count == 3:
                        count += 1
                    if not blocked_diagonal_right and diagonal_right_count == 3:
                        count += 1
                    if not blocked_diagonal_left and diagonal_left_count == 3:
                        count += 1
        if nb_symbol == 0:
            return 0
        return ((count/(nb_symbol))*9/8)

    @staticmethod
    def eval_neurone_3(game, symbol):
        total_distance = 0
        num_occurrences = 0

        for row in range(game.rows):
            for col in range(game.cols):
                if game.board[row][col] == symbol:
                    num_occurrences += 1

        if num_occurrences < 2:
            return 0  # La distance n'a pas de sens avec moins de deux occurrences.

        for row1 in range(game.rows):
            for col1 in range(game.cols):
                if game.board[row1][col1] == symbol:
                    for row2 in range(game.rows):
                        for col2 in range(game.cols):
                            if game.board[row2][col2] == symbol:
                                distance = math.sqrt((row1 - row2) ** 2 + (col1 - col2) ** 2)
                                total_distance += distance

        # Calculer la distance euclidienne moyenne
        average_distance = total_distance / (num_occurrences * (num_occurrences - 1))
        max_distance = math.sqrt(game.rows ** 2 + game.cols ** 2)
        return (max_distance - average_distance) / max_distance

    @staticmethod
    def eval_neurone_4(game, symbol, matrice):
        nb_symbol = 0
        score = 0
        for row in range(game.rows):
            for col in range(game.cols):
                if game.board[row][col] == symbol:
                    nb_symbol += 1
                    score += matrice[row][col]
        return (score/nb_symbol)
        

def Play_against_AI():
    root = tk.Tk()
    game = Puissance4GUI(root, 6, 7, True)
    agent = Agent(game)
    is_first_move = True

    def ia_play(event):
        #agent.play_best_move_V0(5)
        #agent.play_best_move_V1(4, 6)
        agent.play_best_move_test(4, 5)
        game.draw_board()  # Redessiner le tableau après le coup de l'IA

        if game.is_winner("O") or game.is_winner("X"):
            print(f"Joueur {game.current_player} gagne !")
            game.canvas.unbind("<ButtonRelease-1>")
            game.canvas.unbind("<Button-1>")
        elif game.is_draw():
            print("Match nul !")
            game.canvas.unbind("<ButtonRelease-1>")
            game.canvas.unbind("<Button-1>")
        else:
            game.current_player = 'X'  # Mise à jour de la couleur du joueur actuel
            agent.game.current_player = 'X'
            nonlocal is_first_move
            is_first_move = False
            game.canvas.bind("<Button-1>", user_click)  # Rebind pour que le joueur puisse jouer

    def user_click(event):
        nonlocal is_first_move
        col = event.x // game.cell_size
        if is_first_move:
            if not game.is_valid_move(col):
                return
            game.make_move(col)
            game.draw_board()
            is_first_move = False
            game.current_player = 'O'  # Forcer la couleur "O" lorsque l'IA commence
            agent.game.current_player = 'O'
            game.canvas.unbind("<Button-1>")
            game.canvas.bind("<ButtonRelease-1>", ia_play)
        else:
            if not game.is_valid_move(col):
                return
            game.make_move(col)
            game.draw_board()
            game.current_player = 'O'
            agent.game.current_player = 'O'
            if game.is_winner("O") or game.is_winner("X"):
                print(f"Joueur {game.current_player} gagne !")
                game.canvas.unbind("<ButtonRelease-1>")
                game.canvas.unbind("<Button-1>")
            elif game.is_draw():
                print("Match nul !")
                game.canvas.unbind("<ButtonRelease-1>")
                game.canvas.unbind("<Button-1>")
            else:
                game.canvas.bind("<ButtonRelease-1>", ia_play)

    first_player = random.randint(0, 1)
    if first_player == 0:
        is_first_move = True
        # Forcer la couleur "O" lorsque l'IA joue en premier
        game.current_player = 'O'
        agent.game.current_player = 'O'
        ia_play(None)
    else:
        # Laissez l'utilisateur jouer en premier
        game.canvas.bind("<Button-1>", user_click)

    root.mainloop()

def Play_against_AI_good_colors():
    root = tk.Tk()
    game = Puissance4GUI(root, 6, 7, True)
    agent = Agent(game)
    agent.load("poids_1.txt")
    is_first_move = True
    game.current_player = 'X'
    game.draw_board()

    def ia_play(event):
        agent.play_best_move_V0(5)
        #agent.play_best_move_V1(4, 6)
        #agent.play_best_move_test(4, 5)
        game.draw_board()  # Redessiner le tableau après le coup de l'IA

        if game.is_winner("O") or game.is_winner("X"):
            print(f"Joueur {game.current_player} gagne !")
            game.canvas.unbind("<ButtonRelease-1>")
            game.canvas.unbind("<Button-1>")
        elif game.is_draw():
            print("Match nul !")
            game.canvas.unbind("<ButtonRelease-1>")
            game.canvas.unbind("<Button-1>")
        else:
            nonlocal is_first_move
            is_first_move = False
            game.canvas.bind("<Button-1>", user_click)  # Rebind pour que le joueur puisse jouer

    def user_click(event):
        nonlocal is_first_move
        col = event.x // game.cell_size
        if is_first_move:
            if not game.is_valid_move(col):
                return
            game.make_move(col)
            game.draw_board()
            is_first_move = False
            game.canvas.unbind("<Button-1>")
            game.canvas.bind("<ButtonRelease-1>", ia_play)
        else:
            if not game.is_valid_move(col):
                return
            game.make_move(col)
            game.draw_board()
            if game.is_winner("O") or game.is_winner("X"):
                print(f"Joueur {game.current_player} gagne !")
                game.canvas.unbind("<ButtonRelease-1>")
                game.canvas.unbind("<Button-1>")
            elif game.is_draw():
                print("Match nul !")
                game.canvas.unbind("<ButtonRelease-1>")
                game.canvas.unbind("<Button-1>")
            else:
                game.canvas.bind("<ButtonRelease-1>", ia_play)

    first_player = random.randint(0, 1)
    if first_player == 0:
        is_first_move = True
        agent.symbol = 'X'
        ia_play(None)
    else:
        agent.symbol = 'O'
        game.canvas.bind("<Button-1>", user_click)

    root.mainloop()

def IA_against_IA_show():
    root = tk.Tk()
    game = Puissance4GUI(root, 6, 7, True)
    game = Agent.random_game(game, 2)
    agent_1 = Agent(game)
    agent_1.load()
    agent_2 = Agent(game)
    agent_2.load("poids_1.txt")
    game.current_player = 'X'
    

    def ia_1_play():
        agent_1.play_best_move_V0(5)
        #agent_1.play_best_move_V1(6, 6)
        #agent_1.play_best_move_test(4, 5)
        game.draw_board()  
        game.root.update()

        if game.is_winner("O") or game.is_winner("X"):
            print("Agent_1 gagne !")
        elif game.is_draw():
            print("Match nul !")
        else:
            #time.sleep(0.2)
            ia_2_play()

    def ia_2_play():
        agent_2.play_best_move_V0(5)
        #agent_2.play_best_move_V1(4, 6)
        #agent_2.play_best_move_test(4, 5)
        game.draw_board()  
        game.root.update()

        if game.is_winner("O") or game.is_winner("X"):
            print("Agent_2 gagne !")
        elif game.is_draw():
            print("Match nul !")
        else:
            #time.sleep(0.2)
            ia_1_play()

    first_player = random.randint(0, 1)
    if first_player == 0:
        agent_1.symbol = 'X'
        agent_2.symbol = 'O'
        print("agent 1 rouge")
        ia_1_play()
    else:
        agent_1.symbol = 'O'
        agent_2.symbol = 'X'
        print("agent 2 rouge")
        ia_2_play()

    root.mainloop()

def Play():
    root = tk.Tk()
    game = Puissance4GUI(root, 6, 7, True)
    agent = Agent(game)

    def user_click(event):
        col = event.x // game.cell_size
        if game.is_valid_move(col):
            game.make_move(col)
            game.draw_board()
            game.change_current_player()
        if game.is_winner("O") or game.is_winner("X"):
            print(f"Joueur {game.current_player} gagne !")
            game.canvas.unbind("<Button-1>")
        elif game.is_draw():
            print("Match nul !")
            game.canvas.unbind("<Button-1>")

    game.canvas.bind("<Button-1>", user_click)
    root.mainloop()



#  test
def test():
    Agent.training()



def test_best_of():
    Agent.generation = 12
    print(Agent.Best_of(9))
    Agent.generation = 158
    print(Agent.Best_of(9))

def afficher_matrice(matrice):
    for ligne in matrice:
        for valeur in ligne:
            print(f"{valeur:.3f}", end="\t")
        print()

def test_matrice():
    root = tk.Tk()
    game = Puissance4GUI(root, 6, 7, True)
    ne = Neurone(game)
    afficher_matrice(ne.matrice)

def test_save():
    root = tk.Tk()
    game = Puissance4GUI(root, 6, 7, True)
    ag1 = Agent(game)
    ag1.save("poids_1.txt")

def test_load():
    root = tk.Tk()
    game = Puissance4GUI(root, 6, 7, True)
    ag1 = Agent(game)
    ag1.load()
    ag1.Print_poids()
    print("generation :", Agent.generation)

def test_mix_gen():
    root = tk.Tk()
    game = Puissance4GUI(root, 6, 7, True)
    ag1 = Agent(game)
    root = tk.Tk()
    game = Puissance4GUI(root, 6, 7, True)
    ag1 = Agent(game)
    ag2 = Agent(game)
    agent = Agent(game)
    ag1.load("poids.txt")
    ag2.load("poids_1.txt")
    agent.mix_gen(ag1, ag2)
    agent.Print_poids()

def test_gen_change():
    root = tk.Tk()
    game = Puissance4GUI(root, 6, 7, True)
    ag1 = Agent(game)
    ag1.Print_poids()
    ag1.gen_change(2, 0.5, True)
    ag1.Print_poids()

def test_match_of_IA():
    root = tk.Tk()
    game = Puissance4GUI(root, 6, 7, True)
    ag_poids = Agent(game)
    ag_poids.load()
    ag_moi = Agent(game)
    ag_moi.load("poids_1.txt")
    if Agent.match_of_IA(ag_poids, ag_moi, 199, 2) == ag_poids:
        print("poids gagne")
    else:
        print("moi gagne")

if __name__ == '__main__':
    #Play_against_AI()
    #Play_against_AI_good_colors()
    #IA_against_IA_show()
    #Play()
    #test_matrice()
    test_match_of_IA()
    #test_save()
    #test_load()
    #test_mix_gen()
    #test_gen_change()
    #test_best_of()

    #test()

    #  test de temps IA_againt_IA
    '''temp_1 = time.time()
    for i in range(20):
        test()
    temp_2 = time.time()
    temps_moy = (temp_2 - temp_1)/20
    print(f"Temps total écoulé : {temps_moy} secondes.")'''

    

