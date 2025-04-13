from django.shortcuts import render, redirect
from django.views import View
from .forms import RecipyForm
from .langchain import get_gemini_response



class Home(View):
    def get(self, request):
        form = RecipyForm()
        return render(request, 'cheftemplate/home.html', {'form': form, 'response': None})
    
    def post(self, request):
        form = RecipyForm(request.POST)
        if form.is_valid():
            recipy_message = form.cleaned_data['recipymessaage']
            # Get the response from the Gemini API
            response_text = get_gemini_response(recipy_message)
            return render(request, 'cheftemplate/home.html', {'form': form, 'response': response_text})
        else:
            return render(request, 'cheftemplate/home.html', {'form': form, 'response': None})
