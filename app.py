from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "uploads"
CHART_FOLDER = "static/charts"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)

df = None  # dataset stored in memory after upload
users = {"admin": "admin123"}  # demo user


# ------------------ LOGIN & REGISTER ------------------ #
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["username"]
        pwd = request.form["password"]
        if uname in users and users[uname] == pwd:
            session["user"] = uname
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password", "error")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form["username"]
        pwd = request.form["password"]
        if uname in users:
            flash("User already exists", "error")
        else:
            users[uname] = pwd
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ------------------ DASHBOARD ------------------ #
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    global df
    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        # Handle dataset upload
        if "dataset" in request.files:
            file = request.files["dataset"]
            if file and file.filename != "":
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)

                try:
                    dftmp = pd.read_csv(filepath)
                    dftmp.columns = dftmp.columns.str.strip().str.lower()
                    required = {"rollno", "year", "cgpa", "attendance", "projects", "activities"}
                    if not required.issubset(set(dftmp.columns)):
                        missing = required - set(dftmp.columns)
                        flash(f"Dataset missing columns: {', '.join(missing)}", "error")
                    else:
                        dftmp["rollno"] = dftmp["rollno"].astype(str).str.strip()
                        dftmp["year"] = pd.to_numeric(dftmp["year"], errors="coerce")
                        for c in ["cgpa", "attendance", "projects", "activities"]:
                            dftmp[c] = pd.to_numeric(dftmp[c], errors="coerce").fillna(0)
                        df = dftmp.copy()
                        flash("Dataset uploaded successfully!", "success")
                except Exception as e:
                    flash(f"Error reading dataset: {e}", "error")

    table_html = df.head().to_html(classes="table table-striped") if df is not None else None
    return render_template(
        "dashboard.html",
        user=session.get("user"),
        table_html=table_html,
        selected_algorithm=session.get("algorithm")
    )


# ------------------ SAVE SELECTED ALGORITHM ------------------ #
@app.route("/select_algorithm", methods=["POST"])
def select_algorithm():
    algo = request.form.get("algorithm")
    if algo:
        session["algorithm"] = algo
        flash(f"Algorithm '{algo.replace('_',' ').title()}' selected!", "success")
    else:
        flash("Please select a valid algorithm.", "error")
    return redirect(url_for("dashboard"))


# ------------------ HELPERS ------------------ #
def clear_charts():
    for f in os.listdir(CHART_FOLDER):
        try:
            os.remove(os.path.join(CHART_FOLDER, f))
        except Exception:
            pass


def calc_performance_series(df_sub):
    cgpa_pct = (df_sub["cgpa"] / 10.0) * 100
    attendance_pct = df_sub["attendance"]
    proj_max = df_sub["projects"].max() if df_sub["projects"].max() > 0 else 1
    act_max = df_sub["activities"].max() if df_sub["activities"].max() > 0 else 1
    projects_pct = (df_sub["projects"] / proj_max) * 100
    activities_pct = (df_sub["activities"] / act_max) * 100
    perf = pd.concat([cgpa_pct, attendance_pct, projects_pct, activities_pct], axis=1).mean(axis=1)
    return perf


# ------------------ VISUALIZE HOME ------------------ #
@app.route("/visualize")
def visualize():
    global df
    if "user" not in session:
        return redirect(url_for("login"))
    if df is None:
        flash("Please upload a dataset first from Dashboard.", "error")
        return redirect(url_for("dashboard"))
    rollnos = df["rollno"].unique().tolist()
    return render_template("visualize.html", user=session.get("user"),
                           rollnos=rollnos, selected=None, mode=None,
                           chart=None, top_students=None)


# ------------------ ALL STUDENTS ------------------ #
@app.route("/visualize/all/<metric>")
def visualize_all(metric):
    global df
    if df is None:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("dashboard"))

    if metric not in {"cgpa", "attendance", "projects", "activities", "performance"}:
        flash("Invalid metric", "error")
        return redirect(url_for("visualize"))

    clear_charts()
    df_local = df.copy()
    grouped = df_local.groupby("year").mean(numeric_only=True)

    if metric == "performance":
        perf_series = calc_performance_series(df_local)
        df_local["performance"] = perf_series
        series_to_plot = df_local.groupby("year")["performance"].mean()
    else:
        series_to_plot = grouped[metric]

    filename = f"all_{metric}.png"
    plt.figure(figsize=(8, 5))
    series_to_plot.plot(marker="o")
    plt.title(f"Average {metric.capitalize()} by Year (All Students)")
    plt.xlabel("Year")
    plt.ylabel(metric.capitalize())
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_FOLDER, filename))
    plt.close()

    rollnos = df["rollno"].unique().tolist()
    return render_template("visualize.html", rollnos=rollnos, selected=None,
                           mode="all", chart=f"charts/{filename}", top_students=None)


# ------------------ SINGLE STUDENT ------------------ #
@app.route("/visualize/student", methods=["POST", "GET"])
def visualize_student():
    global df
    if df is None:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("dashboard"))

    rollnos = df["rollno"].unique().tolist()
    if request.method == "POST":
        rollno = str(request.form.get("rollno")).strip()
        if rollno == "":
            flash("Please select a roll number.", "error")
            return redirect(url_for("visualize"))
        return redirect(url_for("visualize_student_metric", metric="cgpa", rollno=rollno))

    return render_template("visualize.html", rollnos=rollnos, selected=None,
                           mode="single", chart=None, top_students=None)


@app.route("/visualize/student/<metric>/<rollno>")
def visualize_student_metric(metric, rollno):
    global df
    if df is None:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("dashboard"))

    student_df = df[df["rollno"] == str(rollno)].copy()
    if student_df.empty:
        flash(f"No data for Roll No {rollno}", "error")
        return redirect(url_for("visualize"))

    if metric == "performance":
        student_df["performance"] = calc_performance_series(student_df)
        series_to_plot = student_df.groupby("year")["performance"].mean()
    else:
        series_to_plot = student_df.groupby("year").mean(numeric_only=True)[metric]

    filename = f"student_{rollno}_{metric}.png"
    plt.figure(figsize=(8, 5))
    series_to_plot.plot(marker="o")
    name = student_df.iloc[0]["name"] if "name" in student_df.columns else rollno
    plt.title(f"{metric.capitalize()} trend - {name} ({rollno})")
    plt.xlabel("Year")
    plt.ylabel(metric.capitalize())
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_FOLDER, filename))
    plt.close()

    rollnos = df["rollno"].unique().tolist()
    return render_template("visualize.html", rollnos=rollnos, selected=rollno,
                           mode="single", chart=f"charts/{filename}", top_students=None)


# ------------------ TOP STUDENTS ------------------ #
@app.route("/visualize/top")
def visualize_top():
    global df
    if df is None:
        flash("Please upload dataset first.", "error")
        return redirect(url_for("dashboard"))

    clear_charts()
    grouped = df.groupby(["rollno", "name"]).mean(numeric_only=True).reset_index()
    grouped["performance"] = calc_performance_series(grouped)
    top10 = grouped.sort_values(by="performance", ascending=False).head(10)

    filename = "top10_students.png"
    plt.figure(figsize=(10, 6))
    plt.bar(top10["name"].astype(str), top10["performance"])
    plt.title("Top 10 Students by Performance")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Performance (0-100)")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_FOLDER, filename))
    plt.close()

    rollnos = df["rollno"].unique().tolist()
    top_list = top10[["rollno", "name", "performance"]].to_dict("records")
    return render_template("visualize.html", rollnos=rollnos, selected=None,
                           mode="top", chart=f"charts/{filename}", top_students=top_list)


# ------------------ RUN ALGORITHM ------------------ #
@app.route("/run_algorithm")
def run_algorithm():
    global df
    if df is None:
        flash("Please upload dataset before running algorithm", "error")
        return redirect(url_for("dashboard"))

    algo = session.get("algorithm")
    if not algo:
        flash("Please select an algorithm first", "error")
        return redirect(url_for("dashboard"))

    try:
        df.columns = df.columns.str.strip().str.lower()
        X = df[["attendance", "projects", "activities"]]
        y = df["cgpa"]
    except KeyError:
        flash("Dataset must contain 'attendance', 'projects', 'activities', and 'cgpa' columns", "error")
        return redirect(url_for("dashboard"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algo == "linear_regression":
        model = LinearRegression()
    elif algo == "decision_tree":
        model = DecisionTreeRegressor(random_state=42)
    elif algo == "random_forest":
        model = RandomForestRegressor(random_state=42)
    elif algo == "knn":
        model = KNeighborsRegressor()
    elif algo == "svm":
        model = SVR()
    else:
        flash("Unknown algorithm selected", "error")
        return redirect(url_for("dashboard"))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    flash(f"{algo.replace('_',' ').title()} Model -> MSE: {mse:.2f}, R²: {r2:.2f}", "success")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(debug=True)
