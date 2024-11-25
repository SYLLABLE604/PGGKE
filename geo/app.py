from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
from geo import main

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/generate', methods=['POST'])
# def generate():
#     json_file = request.files.get('json_file')
#     csv_file = request.files.get('csv_file')

#     if json_file and csv_file:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         json_filename = f"edge_{timestamp}.csv"
#         csv_filename = f"node_{timestamp}.csv"
#         json_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
#         csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
#         json_file.save(json_path)
#         csv_file.save(csv_path)

#         # 调用 main 函数，传入用户上传的文件路径
#         connections_data, alternative_data = main(json_path, csv_path)

#         # 返回数据给前端
#         return jsonify({
#             'connections': connections_data,
#             'alternative': alternative_data
#         })
#     else:
#         return jsonify({'error': 'Files not provided'}), 400

@app.route('/generate', methods=['POST'])
def generate():
    json_file = request.files.get('json_file')
    csv_file = request.files.get('csv_file')

    if json_file and csv_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"edge_{timestamp}.csv"
        csv_filename = f"node_{timestamp}.csv"
        json_path = os.path.join(app.config['UPLOAD_FOLDER'], json_filename)
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
        json_file.save(json_path)
        csv_file.save(csv_path)

        # 调用 main 函数，获取 JSON 数据
        connections_data, alternative_data = main(json_path, csv_path)

        # 返回数据给前端
        return jsonify({
            'connections': connections_data,
            'alternative': alternative_data
        })
    else:
        return jsonify({'error': 'Files not provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
