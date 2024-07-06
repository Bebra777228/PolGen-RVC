import gradio as gr

def swap_visibility():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=''), gr.update(value=None)
    

def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo in ['mangio-crepe']:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def update_f0_method(use_hybrid_methods):
    if use_hybrid_methods:
        return gr.Dropdown.update(choices=['hybrid[rmvpe+fcpe]', 'hybrid[rmvpe+crepe]', 'hybrid[crepe+rmvpe]', 'hybrid[crepe+fcpe]', 'hybrid[crepe+rmvpe+fcpe]'], value='hybrid[rmvpe+fcpe]')
    else:
        return gr.Dropdown.update(choices=['rmvpe+', 'fcpe', 'rmvpe', 'mangio-crepe', 'crepe', 'harvest', 'dio', 'pm'], value='rmvpe+')
