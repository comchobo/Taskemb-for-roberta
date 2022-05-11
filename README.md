# Taskemb-for-roberta

This code is edited version of Taskemb, written by Vu et al. (https://github.com/tuvuumass/task-transferability)
I made this because the original uses old version of Transformers and customized one. And It's not easy to use on Roberta or other models.
So I editted it slightly.

The original code uses retain_grad() function in Transformers==2.1.1, which I couldn't see on recent dependency (maybe it was deprecated?).
So I customized modeling_roberta.py in Transformers like
``` 
class RobertaOutput(nn.Module):

...

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```
into

``` 
class RobertaOutput(nn.Module):

...

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
->      self.layer_output = hidden_states
->      self.layer_output.retain_grad()
        return hidden_states
```
Instead of editting those on your own, you can use my customized file. (You have to download modeling_roberta.py and move it into python3.x/site-packages/transformers/models/roberta/)
It can be applicated into various models. You can customize those editting 'OOOSelfAttention, OOOOutput, OOOForSequenceClassification' functions (or other functions if you want to apply this in other tasks).
