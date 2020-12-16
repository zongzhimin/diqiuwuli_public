import obspy

st = obspy.read("data/example.mseed")
st.write("temp.mseed", format="mseed")