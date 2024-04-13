from flask import Flask, render_template, request
import pandas as pd
import helper
import json
import plotly
import pickle
import qwerty

app = Flask(__name__)
unique_states = "karnataka"
unique_cities = "Bangalore"
loc_list = []
# restaurant_name1 = sorted(df['res_name'].unique())
# restaurant_name2 = sorted(df['res_name'].unique())

# df = pd.read_excel('data.xlsx')
#
#
#
#
#
# loc_list = df['location'].unique()
#
# print("loc_list")
# print(loc_list)
# # locality = st.selectbox('Select Locality', loc_list)
# df4 = df[df.location == "Residency Road"]
#
# print(df4)
# df5 = df4.reset_index(drop=True)
# print("df5")
# print(df5)
# loc_count = df[df.location == "Residency Road"]['location'].value_counts()
# fig = helper.locality_count(loc_count)
# # st.plotly_chart(fig)
#
#
# df_loc = pd.read_csv("locations.csv")
#
# lat = [i for i in df_loc['latitude']]
# # print("lat" + lat)
# long = [j for j in df_loc['longitude']]
# # print("long" + long)
#
# map2 = helper.mul_rest(df5, lat, long)
# st.subheader('Restaurant in ' + locality + ' on basis of price range')
# st_folium(map2, width=800, height=550)
# st.subheader('Restaurant in ' + locality + ' on basis of Rating')
# map2 = helper.mul_rest_rating(df5, lat, long)

# st_folium(map2, width=800, height=550)
# st_folium(map2, width=800, height=550)


# loc_list = df4['locality'].unique()

dfloc = pd.read_csv('locations.csv')

df_loc = dfloc["name"]


# loc_list = df4['locality'].unique()

# print(loc_list)
# print("loc_list")


# Read the CSV file
df = pd.read_excel('data.xlsx')


# Extract unique state, city, and restaurant names and sort them
unique_states = "karnataka"
unique_cities = "Bangalore"
restaurant_name1 = sorted(df['res_name'].unique())
restaurant_name2 = sorted(df['res_name'].unique())



# Load the model
model = pickle.load(open('model_pred.pkl', 'rb'))
main_df = pd.read_excel("data.xlsx")
# loc_list  = df["location"].unique()

loc_list = ""

dfz = pd.read_csv("final.csv")

state = "Karnataka"
city = "Bangalore"
# Define routes
# @app.route('/')
# def trial():
#     return render_template('index.html')



@app.route('/restaurant_comparison', methods=['GET', 'POST'])
def restaurant_comparison():
    global restaurant_name1
    if request.method == 'POST':
        selected_state = request.form['state']
        selected_city = request.form['city']
        rest_1 = request.form['restaurant1']
        rest_2 = request.form['restaurant2']

        rest1 = df[df.res_name == rest_1]['id']
        if not rest1.empty:
            rest1 = rest1.reset_index(drop=True)[0]
        else:
            # Handle case when no matching restaurant is found
            return "Restaurant 1 not found."

        rest2 = df[df.res_name == rest_2]['id']
        if not rest2.empty:
            rest2 = rest2.reset_index(drop=True)[0]
        else:
            # Handle case when no matching restaurant is found
            return "Restaurant 2 not found."

        fig1 = helper.res_rating_com(df, rest1, rest2)
        fig2 = helper.res_cost_com(df, rest1, rest2)

        n1 = list(df[df.id == rest1]['res_name'])[0]
        n2 = list(df[df.id == rest2]['res_name'])[0]

        graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
        graphJSON2 = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('restaurant_comparison.html',
                               graphJSON1=graphJSON1,
                               graphJSON2=graphJSON2,
                               restaurant1=n1,
                               restaurant2=n2,
                               selected_state=selected_state,
                               selected_city=selected_city,
                               selected_restaurant1=rest_1,
                               selected_restaurant2=rest_2,
                               restaurant_name1 = restaurant_name1)

    else:
        # Render the form with the default values
        return render_template('restaurant_comparison.html' ,restaurant_name1 = restaurant_name1)


@app.route('/restaurant_on_map', methods=['GET', 'POST'])
def restaurant_on_map():
    df = pd.read_csv("final.csv")
    df4 = df[(df.state == "Karnataka") & (df.city == "Bangalore")]

    # Correct the assignment operator
    global loc_list
    loc_list = df4['locality'].unique()

    if request.method == 'POST':
        locality = request.form["loc"]

        df4 = df4[df4.locality == locality]
        df5 = df4.reset_index(drop=True)

        lat = [i for i in df5['latitude']]
        long = [j for j in df5['longitude']]
        map1 = helper.mul_rest(df5, lat, long)
        map2 = helper.mul_rest_rating(df5, lat, long)

        if map1 is not None and map2 is not None:
            map1_html = map1._repr_html_()
            map2_html = map2._repr_html_()

            return render_template('restaurant_on_map.html',
                                   map1_html=map1_html,
                                   map2_html=map2_html,
                                   locality=locality,
                                   loc_list=loc_list)  # Pass loc_list to the template
    else:
        return render_template('restaurant_on_map.html',
                               unique_states="Karnataka",  # Assuming these are constants or global variables
                               unique_cities="Bangalore",
                               loc_list=loc_list)  # Pass loc_list to the template


@app.route('/', methods=["GET", "POST"])
def restaurant_recommendation():
    if request.method == "POST":
        cuisines = request.form.get("cuisines")
        current_location = request.form.get("currentlocation")
        location = request.form.get("location")
        dishcategory = request.form.get("dishcategory")
        res_name = qwerty.abc(cuisines, location, current_location, dishcategory)

        # Create a list of dictionaries with the necessary data
        res_data = []
        for res in res_name:
            row = main_df.loc[main_df['res_name'] == res]
            if not row.empty:
                res_data.append({
                    'res_name': res,
                    'rating': row['rating'].values[0],
                    'cusines': row['cusines'].values[0],
                    'res_link': row['res_link'].values[0]
                })

        # Redirect back to the form with the submitted values
        return render_template("restaurant_recommendation.html", res_data=res_data, cuisines=cuisines, current_location=current_location, location=location, dishcategory=dishcategory)
    return render_template("restaurant_recommendation.html")
if __name__ == '__main__':
    app.run(debug=True)