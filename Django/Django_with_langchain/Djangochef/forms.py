from django import forms

class RecipyForm(forms.Form):
    recipymessaage = forms.CharField(max_length=255, 
                            widget=forms.TextInput(
                                attrs={'placeholder' : 'Ask you recipe'}))