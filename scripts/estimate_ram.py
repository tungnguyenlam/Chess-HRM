import chess
from pympler import asizeof


def estimate():
    # 1. Raw JSON record (typical size for 1 position)
    sample_record = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "history": ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * 7,
        "move": "e2e4",
        "cp": 15,
        "depth": 18,
        "white_elo": 2500,
        "black_elo": 2450,
    }
    json_size = asizeof.asizeof(sample_record)

    # 2. Python-chess Board object
    board = chess.Board()
    board_obj_size = asizeof.asizeof(board)

    # 3. Encoded Board Tensor [8, 8, 119] in float32
    # 8 * 8 * 119 * 4 bytes
    tensor_size = 8 * 8 * 119 * 4

    # 4. Total per move
    total_per_move = json_size + board_obj_size + tensor_size

    print("--- Per Move Estimates ---")
    print(f"JSON Record: {json_size / 1024:.2f} KB")
    print(f"Board Object: {board_obj_size / 1024:.2f} KB")
    print(f"Tensor [8,8,119]: {tensor_size / 1024:.2f} KB")
    print(f"Total: {total_per_move / 1024:.2f} KB")

    # 5. Shard Estimate (100,000 games, ~40 moves per game)
    total_moves = 100000 * 40
    shard_ram = (total_moves * total_per_move) / (1024**3)

    print("\n--- 100,000 Game Shard (Streaming) ---")
    print(f"Moves in Shard: {total_moves:,}")
    print(f"RAM if all moves held: {shard_ram:.2f} GB")

    # 6. Streaming Buffer Estimate (buffer_size=10,000)
    buffer_ram = (10000 * total_per_move) / (1024**2)
    print("\n--- Runtime RAM (buffer_size=10,000) ---")
    print(f"Active Buffer RAM: {buffer_ram:.2f} MB")


if __name__ == "__main__":
    estimate()
