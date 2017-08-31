#!/usr/bin/python3
# Fuzz testing for UCI engines using python-chess.

import chess
import chess.uci
import random
import logging
import sys

random.seed(123456)

TEST_ROUNDS = 10000
MAX_PIECES = 32

#logging.basicConfig(level=logging.DEBUG)

def random_piece_list(max_pieces=64):
    COLORS = [chess.WHITE, chess.BLACK]
    PIECE_TYPES = [None, chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

    pieces = []
    pieces += [chess.Piece(chess.KING, chess.WHITE)]
    pieces += [chess.Piece(chess.KING, chess.BLACK)]

    random.shuffle(PIECE_TYPES)
    for piece_type in PIECE_TYPES:
        random.shuffle(COLORS)
        for color in COLORS:
            piece = chess.Piece(piece_type, color) if piece_type else None
            pieces += [piece] * random.randint(0, max_pieces - len(pieces))

    pieces += [None] * (64 - len(pieces))

    random.shuffle(pieces)
    return pieces

def random_positions(max_pieces=64):
    board = chess.Board()

    while True:
        board.clear()
        piece_list = random_piece_list(max_pieces)
        for square, piece in enumerate(piece_list):
            if piece:
                board.set_piece_at(square, piece)

        board.turn = random.choice(chess.COLORS)

        # Skip positions with the opposite side in check.
        if board.was_into_check():
            continue

        # Skip positions with pawns on the promotion rank.
        if board.pawns & board.occupied_co[chess.WHITE] & chess.BB_RANK_8:
            continue
        if board.pawns & board.occupied_co[chess.BLACK] & chess.BB_RANK_1:
            continue

        yield board

        # Generate positions with kinda valid en-passant squares.
        if board.turn == chess.BLACK:
            for potential_ep in chess.SquareSet(chess.BB_RANK_3):
                board.ep_square = potential_ep
                if (board.piece_at(chess.square(chess.file_index(potential_ep), 3)) == chess.Piece(chess.PAWN, chess.WHITE)
                        and not board.piece_at(chess.square(chess.file_index(potential_ep), 2))
                        and not board.piece_at(chess.square(chess.file_index(potential_ep), 1))):
                    yield board
        else:
            for potential_ep in chess.SquareSet(chess.BB_RANK_6):
                board.ep_square = potential_ep
                if (board.piece_at(chess.square(chess.file_index(potential_ep), 4)) == chess.Piece(chess.PAWN, chess.BLACK)
                        and not board.piece_at(chess.square(chess.file_index(potential_ep), 5))
                        and not board.piece_at(chess.square(chess.file_index(potential_ep), 6))):
                    yield board

if __name__ == "__main__":
    engine = chess.uci.popen_engine("/home/sukrit/stockfish-8-linux/Linux/stockfish_8_x64")

    generator = random_positions(MAX_PIECES)

    for _ in range(TEST_ROUNDS):
        board = next(generator)

        engine.ucinewgame()
        engine.position(board)
        print(board.shredder_fen())
        bestmove, pondermove = engine.go(movetime=20)

    engine.quit()