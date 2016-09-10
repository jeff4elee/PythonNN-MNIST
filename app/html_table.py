from flask_table import Table, Col

class DataTable(Table):
    nickname = Col('Name')
    score = Col('Score')
    img_src = Col('Image Uploaded')
