from multiapp import MultiApp

from apps import home
from apps import nonlinearfilter_main
from apps.morphological import morphological_main
from apps.arithmetic_operations import arithmetic_main

app = MultiApp()


# option = st.selectbox(
#     'Select from the options',
#     ('Home', 'Filters', 'Doc scanner','add text'), key = 1)


# if(option=='Filters'):
#     opt = st.selectbox(
#     'Select from the options',
#     ('sepia', 'Filter1', 'filter2','filter3'), key = 2)

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Morphological", morphological_main.app)
app.add_app("Arithmetic", arithmetic_main.app)
app.add_app("Non-linear filter" , nonlinearfilter_main.app)

# The main app
app.run()