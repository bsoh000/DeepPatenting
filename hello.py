from flask import Flask

from flask import request

app = Flask(__name__)

@app.route('/')
def index():
	user_agent = request.headers.get('User-Agent')
	return '<p>Your browser is %s</p>' % user_agent
	# return '<h1>Hello World!</h1>'
	
@app.route('/user/<dir>')
def user(dir):
	return '<h1>Hello, %s!</h1>' % dir

if __name__ == '__main__':
	app.run(debug=True)