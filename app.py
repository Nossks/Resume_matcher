from flask import Flask ,request ,render_template
from utils import get_recomendation , resume_preprocess
import fitz
from io import BytesIO

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def landing_page():
    if request.method=="POST":
        text=""
        if "resume_file" in request.files and request.files["resume_file"].filename!="":
            file = request.files["resume_file"]
            pdf = fitz.open(stream=BytesIO(file.read()))
            for page in pdf:
                text += page.get_text()
        elif "resume_text" in request.form and request.form["resume_text"].strip()!="":
            text = request.form["resume_text"]
        if not text.strip():
            return render_template("index.html",recommendation=["No resume data provided"])
        res = get_recomendation(resume_preprocess(text))
        return render_template("index.html",recommendation=res)
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)

