
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import datetime
import os

app = Flask(__name__)
# [DB_TYPE]+[DB_CONNECTOR]://[USERNAME]:[PASSWORD]@[HOST]:[PORT]/[DB_NAME]
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('SQLALCHEMY_DATABASE_URI')
db = SQLAlchemy(app) # pip3 install psycopg2-binary

class userRatings(db.Model):
	__tablename__ = 'userRatings'
	__table_args__ = {'keep_existing': True}
	id = db.Column(db.Integer, primary_key = True)
	uid = db.Column(db.String(40))
	movie = db.Column(db.String(6))
	rating = db.Column(db.Integer)
	timestamp = db.Column(db.TIMESTAMP, default = datetime.datetime.now)

	def __init__(self, userId, movieId, rating):
		# 不assign self.id,self.timestamp時, 兩者便會自動使用預設值
		self.uid = userId
		self.movie = movieId
		self.rating = rating

	def addRating(userId, movieId, rating):
		with app.app_context():
			row = userRatings(userId, movieId, rating)
			db.session.add(row)
			db.session.commit()

	# 更新紀錄檔
	def Record_adder(userId, movieId, rating):
		with app.app_context():
			userRatings.addRating(userId, movieId, rating)
	
	# 讀取紀錄檔
	def Record_reader(userId):
		with app.app_context():
			records = userRatings.query.filter_by(uid = userId).all()
		return records
