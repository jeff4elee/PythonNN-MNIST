from app import db

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nickname = db.Column(db.String(64), index=True, unique=True)
    score = db.Column(db.Integer, index=True)
    img_src = db.Column(db.String)

    def __init__(self, nickname, score, img_src):
        self.nickname = nickname
        self.score = score
        self.img_src = img_src
    
    def __repr__(self):
        return '<User %r>' % (self.nickname)

