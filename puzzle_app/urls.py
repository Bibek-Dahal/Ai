from django.urls import path
from . import views

urlpatterns = [
    path('', views.render_state_space_tree, name='state_space_tree'),
    path('dfs',views.renderDfs,name="dfs"),
    path('man',views.manhatten,name="man")
]
