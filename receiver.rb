# Vision-to-Audio Note Test
# This receives OSC messages from the Python vision system
# and logs/plays the MIDI notes
# currently listens to the volume (which is specified by hand height) --> will
# vary the volume based on the visual input. 

live_loop :receive_volume do
  use_real_time
  v = sync "/osc*/volume"
  set :vol, v[0]
end

# gesture type to sound
live_loop :receive_sound do
  use_real_time
  s = sync "/osc*/sound"
  set :current_sound, s[0]
  puts "Gesture changed to: #{s[0]}"
end

# play note/pitch when received
live_loop :receive_note do
  use_real_time
  n = sync "/osc*/note"
  note_num = n[0]
  
  # current volume and sound type
  vol = get[:vol] || 0.5
  sound_type = get[:current_sound] || 0
  
  # log received note
  puts "Received note: #{note_num}"
  
  # play gesture to sound
  case sound_type
  when 0  # open palm - piano
    use_synth :piano
    play note_num, amp: vol, release: 1.5
    puts "Playing PIANO"
    
  when 1  # fist - deep bass
    use_synth :bass_foundation
    play note_num - 24, amp: vol * 1.2, release: 1.0
    puts "Playing BASS"
    
  when 2  # peace sign - synth lead
    use_synth :prophet
    play note_num, amp: vol * 0.8, release: 0.5, cutoff: 100
    puts "Playing SYNTH"
    
  when 3  # thumbs up - bell
    use_synth :pretty_bell
    play note_num + 12, amp: vol * 0.7, release: 2.0
    puts "Playing BELL"
    
  when 4  # pointing - flute
    use_synth :blade
    play note_num + 7, amp: vol * 0.6, release: 1.0
    puts "Playing FLUTE"
  end
  
  sleep 0.5  # 0.5 = medium, 0.2 = faster, 1.0 = slower
end