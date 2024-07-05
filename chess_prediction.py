import cv2
import chess
import chess.engine as chess_engine
import tkinter as tk
from tkinter import filedialog
import numpy as np
from itertools import groupby
from keras.models import load_model

def ask_user_color():
    color = input("Are you playing as white or black? Ex. black: ").strip().lower()
    if color not in ["white", "black"]:
        print("Invalid choice. Please enter 'white' or 'black'.")
        return ask_user_color()
    return color

def get_image_choice():
    choice = input("Do you want to (u)pload an image or (c)apture an image? ").strip().lower()
    if choice not in ["u", "c"]:
        print("Invalid choice. Please enter 'c' or 'u'.")
        return get_image_choice()
    return choice

def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image. Try again.")
            continue
        cv2.imshow('Capture Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

def upload_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    if not file_path:
        print("No file selected. Please try again.")
        return upload_image()
    image = cv2.imread(file_path)
    return image

def preprocessing_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def warp_chessboard(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            points = approx.reshape(4, 2)
            break
    else:
        raise ValueError("Could not find chessboard corners.")
    
    points = sorted(points, key=lambda x: x[0])
    top_points = sorted(points[:2], key=lambda x: x[1])
    bottom_points = sorted(points[2:], key=lambda x: x[1])
    src = np.array([top_points[0], top_points[1], bottom_points[0], bottom_points[1]], dtype='float32')

    side = max([
        np.linalg.norm(src[0] - src[1]),
        np.linalg.norm(src[1] - src[2]),
        np.linalg.norm(src[2] - src[3]),
        np.linalg.norm(src[3] - src[0]),
    ])

    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (int(side), int(side)))

    return warped, side

def segment_board(warped, side):
    square_size = int(side / 8)
    squares = []

    for y in range(8):
        for x in range(8):
            square = warped[y * square_size:(y + 1) * square_size, x * square_size:(x + 1) * square_size]
            squares.append(square)
    
    return squares

def predict_piece(square_image):
    # Load your pre-trained model (replace 'chess_model.h5' with your actual model path)
    model_path = 'chess_model.h5'
    model = load_model(model_path)

    square_image = cv2.resize(square_image, (64, 64))
    square_image = square_image / 255.0
    square_image = np.expand_dims(square_image, axis=0)
    
    # Predict the piece
    predictions = model.predict(square_image)
    piece = np.argmax(predictions)
    
    # Map the prediction to the corresponding piece
    piece_map = {
        0: '',
        1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
        7: 'p', 8: 'n', 9: 'b', 10: 'r', 11: 'q', 12: 'k'
    }
    return piece_map[piece]

def squares_to_fen(squares):
    board = []
    for i, square in enumerate(squares):
        row, col = divmod(i, 8)
        piece = predict_piece(square)
        board[row].append(piece if piece else '1') if col == 0 else None
    
    # Merge empty squares and format to FEN
    # for i in range(len(board)):
    #     board[i] = ''.join([str(len(list(g))) if k == '1' else k for k, g in groupby(board[i])])
    for i in range(len(board)):
        merged_row = ''
        count = 0
        for char in board[i]:
            if char == '1':
                count += 1
            else:
                if count > 0:
                    merged_row += str(count)
                    count = 0
                merged_row += char
        if count > 0:
            merged_row += str(count)
        board[i] = merged_row
    
    board_state = chess.Board()
    board_state.set_fen('/'.join(board))
    fen = board_state.fen()
    return fen

def suggest_move(board_state, player_color):
    board = chess.Board(board_state)
    with chess_engine.SimpleEngine.popen_uci("D:\Chess Engine ML\stockfish\stockfish-windows-x86-64-avx2.exe") as engine:
        result = engine.play(board, chess_engine.Limit(time=2.0))
    best_move = result.move
    return best_move

def main():
    player_color = ask_user_color()
    print(f"You are playing as {player_color}")

    choice = get_image_choice()
    if choice == 'c':
        print("Press 'q' to capture the image.")
        image = capture_image()
    else:
        image = upload_image()

    processed_image = preprocessing_image(image)
    warped_image, side_length = warp_chessboard(image)
    squares = segment_board(warped_image, side_length)

    fen = squares_to_fen(squares)
    print("Suggested move: ", fen)

if __name__ == "__main__":
    main()