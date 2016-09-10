from flask import render_template, flash, redirect, request, send_from_directory
from app import app
from app import db, models
from PIL import Image
from script import neural_network
import numpy as np
import flask_uploads
import html_table
import os


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

net = neural_network.Network([784, 30, 10])
net = net.SGD(net.training_data, 2, 10, 3.0, test_data=net.test_data)
max_generations = 10
destination = 'none'

@app.route('/', methods=['GET', 'POST'])
def index():
    '''waits for an image file to be submitted, then uses a trained
        neural network to guess the number'''
    if (request.method == 'POST'):
        
        target = os.path.join(APP_ROOT, 'static/')
        
        if not os.path.isdir(target):
            os.mkdir(target)

        file = request.files['file']
        global destination
        imagename = str(file.filename)
        destination = "/".join([target, imagename])
        file.save(destination)
        prediction = predictOnce(file)
        #ALLOW LARGER THAN ONE DIGIT IMAGES
        #TODO
        #prediction = [predictNumber(file)]
        return render_template('result.html',
                               prediction = prediction)
    else:
        return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if (request.method == 'POST'):
        image = Image.open(destination)        
        value = [int(str(request.form['true_val']))]
        image = image.convert('L')
        image = image.resize((28,28), Image.ANTIALIAS)
        i_v = []
        for y in range(image.size[1]):
            for x in range(image.size[0]):
                i_v.append(image.getpixel((x,y))/255.0)
        input_vector = [np.reshape(i_v, (784, 1))]
        data = zip(input_vector, value)          
        net.SGD(data, 1, 1, 3.0)
        #ALLOW LARGER THAN ONE DIGIT IMAGES
        #TODO
        #prediction = [predictNumber(file)]
        return render_template('index.html',
                               thanks = 'Thank you for your input!')
    else:
        return render_template('result.html')
    
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    '''waits for the form to be submitted, then redirects user to the database
        page, were all the input values are stored'''
    if (request.method == 'POST'):
        nickname = request.form['nickname']
        value = request.form['true_val']
        error = '* Field Required'
        if not nickname:
            if (not value or type(int(value)) != int):
                return render_template("upload.html",
                                       name_error = error,
                                       num_error = error)                                     
            return render_template("upload.html",
                                   name_error = error)
        if (not value or (type(int(value)) != int)):
            return render_template("upload.html",
                                   num_error = error)
        
        target = os.path.join(APP_ROOT, 'static/')
        
        if not os.path.isdir(target):
            os.mkdir(target)

        file = request.files['file']

        filename = file.filename
        destination = "/".join([target, filename])
        file.save(destination)
        score = max_generations-predictUntilCorrect(file, value)
        if(score > 0):
            #models.User.query.delete()
            user = models.User(str(nickname), int(score), img_src=str(filename))
            db.session.add(user)
            db.session.commit()
            return redirect('/database')
        return redirect('/error')
    else:
        return render_template("upload.html")

@app.route('/error')
def error():
    return render_template("error.html")
    
@app.route('/database')
def database():
    table = models.User.query.all()
    new_table = sorted(table, key=lambda user: user.score, reverse = True)
    total = 0
   # for i in range(len(new_table)):
   #     total += new_table[i].score
   # for i in range(len(new_table)):
   #     table[i].score = new_table[i].score/float(total)
    final_table = html_table.DataTable(new_table)
    return render_template("database.html",
                           table = final_table.__html__())

@app.route('/database/<filename>')
def uploaded_images(filename):
    target = os.path.join(APP_ROOT, 'static/')
    return send_from_directory(target, filename)
    
def predictUntilCorrect(file_name, value):
    '''trains neural net until the value can be guessed correctly'''
    image = Image.open(file_name)
    untrained_net = NN.Network([784, 30, 10])
    prediction = -1
    generations = 0
    while(prediction != int(value) and generations < max_generations):
        generations += 1
        trained_net = untrained_net.SGD(untrained_net.training_data[:(((generations%5)+1)*10000)], 1, 10, 1.0)
        prediction = trained_net.feedImage(image)
        flash(prediction)
    return generations

def predictNumber(file_name):
    '''uses already trained neural net to take up to multiple images and
        find their values'''
    #TODO
    
def predictOnce(file_name):
    '''already trained neural net takes a guess at your image file value'''
    image = Image.open(file_name)
    global net
    prediction = net.feedImage(image)
    return prediction
    #flash(type(prediction))
    #flash(type(value))
    #flash(prediction == int(value))

