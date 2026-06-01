from flask import Flask, render_template, request

from predictor import predict_input

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    user_input = ""

    if request.method == "POST":
        user_input = request.form.get("user_input", "").strip()
        result = predict_input(user_input)

    return render_template(
        "index.html",
        result=result,
        user_input=user_input,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
