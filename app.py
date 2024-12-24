from flask import Flask, render_template, request, jsonify, session
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

app = Flask(__name__)
app.secret_key = 'sudoku-solver-secret-key'
MODEL_PATH = 'model/sudoku_model.h5'
model = load_model(MODEL_PATH)


def next_box(quiz):
    for row in range(9):
        for col in range(9):
            if quiz[row][col] == 0:
                return (row, col)
    return False

def possible(quiz, row, col, n):
   
    for i in range(9):
        if quiz[row][i] == n and i != col:
            return False
    
    
    for i in range(9):
        if quiz[i][col] == n and i != row:
            return False
    
    
    box_x = (col // 3) * 3
    box_y = (row // 3) * 3
    
    for i in range(3):
        for j in range(3):
            if quiz[box_y + i][box_x + j] == n and (box_y + i != row or box_x + j != col):
                return False
    
    return True

def is_valid_sudoku(grid):
   
    for row in range(9):
        seen = set()
        for col in range(9):
            num = grid[row][col]
            if num != 0:
                if num in seen:
                    return False
                seen.add(num)
    
    
    for col in range(9):
        seen = set()
        for row in range(9):
            num = grid[row][col]
            if num != 0:
                if num in seen:
                    return False
                seen.add(num)
    
   
    for box_y in range(0, 9, 3):
        for box_x in range(0, 9, 3):
            seen = set()
            for i in range(3):
                for j in range(3):
                    num = grid[box_y + i][box_x + j]
                    if num != 0:
                        if num in seen:
                            return False
                        seen.add(num)
    
    return True

def solve(quiz):
    val = next_box(quiz)
    if val is False:
        return True
    else:
        row, col = val
        for n in range(1,10):
            if possible(quiz, row, col, n):
                quiz[row][col] = n
                if solve(quiz):
                    return True 
                quiz[row][col] = 0
        return False


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 6)
    threshold_img = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return threshold_img

def main_outline(contour):
    biggest = np.array([])
    max_area = 0
    for i in contour:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4,1,2), dtype=np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new

def splitcells(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes

def CropCell(cells):
    Cells_croped = []
    for image in cells:
        img = np.array(image)
        img = img[4:46, 6:46]
        img = Image.fromarray(img)
        Cells_croped.append(img)
    return Cells_croped

def read_cells(cell, model):
    result = []
    for image in cell:
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (32, 32))
        img = img / 255
        img = img.reshape(1, 32, 32, 1)
        
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=1)
        probabilityValue = np.amax(predictions)
        
        if probabilityValue > 0.65:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_numbers():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    
    img_path = "static/uploads/temp.jpg"
    file.save(img_path)
    
    
    image = cv2.imread(img_path)
    image = cv2.resize(image, (450,450))
    
    
    processed = preprocess(image)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest, _ = main_outline(contours)
    
    if biggest.size != 0:
        biggest = reframe(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imagewrap = cv2.warpPerspective(image, matrix, (450,450))
        imagewrap = cv2.cvtColor(imagewrap, cv2.COLOR_BGR2GRAY)
        
        
        cells = splitcells(imagewrap)
        cells_cropped = CropCell(cells)
        numbers = read_cells(cells_cropped, model)
        
       
        grid = np.array(numbers).reshape(9,9).tolist()
        
        
        session['detected_grid'] = grid
        
        return render_template('sudoku.html', grid=grid, detected=True, solved=False)
    
    return jsonify({'error': 'Sudoku ızgarası tespit edilemedi'}), 400

@app.route('/solve', methods=['POST'])
def solve_sudoku():
    grid = session.get('detected_grid')
    if not grid:
        return jsonify({'error': 'Önce bir Sudoku yüklemelisiniz'}), 400
    
    grid = np.array(grid)
    
    
    if not is_valid_sudoku(grid):
        
        error_details = []
        
        
        for row in range(9):
            seen = set()
            for col in range(9):
                num = grid[row][col]
                if num != 0:
                    if num in seen:
                        error_details.append(f"Satır {row+1}'de {num} tekrar ediyor")
                    seen.add(num)
        
        
        for col in range(9):
            seen = set()
            for row in range(9):
                num = grid[row][col]
                if num != 0:
                    if num in seen:
                        error_details.append(f"Sütun {col+1}'de {num} tekrar ediyor")
                    seen.add(num)
        
        
        for box_y in range(0, 9, 3):
            for box_x in range(0, 9, 3):
                seen = set()
                for i in range(3):
                    for j in range(3):
                        num = grid[box_y + i][box_x + j]
                        if num != 0:
                            if num in seen:
                                error_details.append(f"({box_y//3 + 1},{box_x//3 + 1}) kutusunda {num} tekrar ediyor")
                            seen.add(num)
        
        return jsonify({
            'error': 'Girilen Sudoku geçerli değil. Lütfen rakamları kontrol edin.',
            'details': error_details
        }), 400
    
    original_grid = grid.copy()
    if solve(grid):
        return render_template('sudoku.html', grid=grid.tolist(), detected=True, solved=True)
    else:
        return jsonify({
            'error': 'Sudoku çözülemedi.',
            'details': 'Girilen Sudoku çözülebilir değil veya çok karmaşık.'
        }), 400

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)