from transformers import AutoModelForCausalLM
import torch
from anticipation.sample import generate
from anticipation.convert import events_to_midi, midi_to_events
from anticipation import ops
from anticipation.tokenize import extract_instruments

# Check if MPS is available (for M1/M2 Macs)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Load the pretrained model
model = AutoModelForCausalLM.from_pretrained('stanford-crfm/music-medium-800k').to(device)




# Generate a new piece of music
def generate_unconditioned(length=10, output_file='unconditioned.mid'):
    """
    Generate a new piece of music
    length: time in seconds
    output_file: where to save the MIDI file
    """
    events = generate(model, start_time=0, end_time=length, top_p=0.98)
    mid = events_to_midi(events)
    mid.save(output_file)
    print(f"Generated music saved to {output_file}")

# generate accompaniment given melody as controls
def generate_span_infill(input_midi1, infill_start, infill_end, input_end, output_file='span_infill.mid'):
    
    events = midi_to_events(input_midi1)
    segment = ops.clip(events, 0, input_end)
    segment = ops.translate(segment, -ops.min_time(segment, seconds=False))
    CONTROL_OFFSET = 5

    history = ops.clip(segment, 0, infill_start, clip_duration=False)
    anticipated = [CONTROL_OFFSET + tok for tok in ops.clip(segment, infill_end, input_end, clip_duration=False)]
    inpainted = generate(model, infill_start, infill_end, inputs=input, controls=anticipated, top_p=.95)




def generate_accompaniment(input_midi, 
                         start_time=5,
                         end_time=20,
                         melody_instrument=53,
                         output_file='output_with_accompaniment.mid'):
    """
    Generate an accompaniment for a given MIDI file
    
    Parameters:
    - input_midi: path to input MIDI file
    - start_time: time to start generating from (default: 5 seconds)
    - end_time: time to end generation (default: 20 seconds)
    - melody_instrument: MIDI instrument code for melody (default: 53)
    - output_file: where to save the result
    """
    # Load and prepare the input MIDI
    events = midi_to_events(input_midi)
    
    # Clip the segment we want to work with
    segment = ops.clip(events, 0, 10)
    segment = ops.translate(segment, -ops.min_time(segment, seconds=False))
    
    # Extract the melody
    print("Available instruments:", ops.get_instruments(segment).keys())
    events, melody = extract_instruments(segment, [melody_instrument])
    
    # Create history for the model
    history = ops.clip(events, 0, 5, clip_duration=False)
    
    # Generate the accompaniment
    accompaniment = generate(model, 
                           start_time=0, 
                           end_time=20, 
                           inputs=history, 
                           controls=melody, 
                           top_p=0.95, 
                           debug=False)
    
    # Combine the accompaniment with the melody
    output = ops.clip(ops.combine(accompaniment, melody), 0, 20, clip_duration=True)
    
    # Save the result
    mid = events_to_midi(output)
    mid.save(output_file)
    print(f"Generated accompaniment saved to {output_file}")

if __name__ == "__main__":
    # Example usage:
    # Generate a new 10-second piece
    # generate_new_music(length=10, output_file='new_piece.mid')
    
    generate_span_infill('examples/strawberry.mid', 5, 10, 20)

    # Generate accompaniment for an existing MIDI file
    # generate_accompaniment('control1.mid', output_file='with_accompaniment.mid')