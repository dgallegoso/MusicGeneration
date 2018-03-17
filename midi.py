import mido
import numpy as np


NUM_NOTES = 128
NUM_TIME_SHIFTS = 100
TICK_PER_SECOND = 1000
TICKS_PER_BEAT = 5000
DEFAULT_VELOCITY = 50


def decode_midi_1(midi, resolution):
    decoding = []
    bucket = np.zeros(2*NUM_NOTES)
    time = 0
    for message in midi:
        time += message.time
        if time > resolution:
            numSkips = int(time/resolution)
            time -= resolution * numSkips
            decoding.append(bucket)
            bucket = np.zeros(2*NUM_NOTES)
            for _ in range(numSkips-1):
                decoding.append(np.zeros(2*NUM_NOTES))
        if message.type == 'note_on':
            if message.velocity == 0:
                bucket[NUM_NOTES + message.note] = 1
            else:
                bucket[message.note] = 1
        # if message.type not in ['note_on', 'control_change', 'program_change', 'sequencer_specific', 'set_tempo', 'time_signature', 'sysex', 'polytouch', 'end_of_track']:
        #     print 'Warning: Unknown message type ({}).'.format(message.type)
    decoding.append(bucket)
    return np.array(decoding)


def encode_midi_1(decoding, track, resolution):
    offset = 0
    for row in decoding:
        offset += int(TICK_PER_SECOND*resolution)
        for i in np.where(row == 1)[0]:
            if i < NUM_NOTES:
                track.append(mido.Message('note_on', note=i, velocity=DEFAULT_VELOCITY, time=offset))
            else:
                track.append(mido.Message('note_on', note=i-NUM_NOTES, velocity=0, time=offset))
            offset = 0


def decode_midi_2(midi, resolution):
    decoding = []
    emptyBucket = np.zeros(2*NUM_NOTES + NUM_TIME_SHIFTS)
    bucket = emptyBucket.copy()
    time = 0
    counter = 0
    for message in midi:
        time += message.time
        if time > resolution:
            numSkips = int(time/resolution)
            counter += numSkips
            time -= resolution * numSkips
        if message.type == 'note_on':
            if message.velocity == 0:
                bucket[NUM_NOTES + message.note] = 1
            else:
                bucket[message.note] = 1
        if counter > NUM_TIME_SHIFTS:
            numFullShifts = int(np.ceil(counter/float(NUM_TIME_SHIFTS)))
            for _ in range(numFullShifts-1):
                temp = emptyBucket.copy()
                temp[-2] = 1
                decoding.append(temp)
            counter -= (numFullShifts - 1) * NUM_TIME_SHIFTS
        if np.sum(bucket) > 0 and counter > 0:
            if counter > 1:
                bucket[2*NUM_NOTES + counter - 2] = 1
            decoding.append(bucket)
            bucket = emptyBucket.copy()
            counter = 0
    bucket[2*NUM_NOTES + int(time/resolution)] = 1
    decoding.append(bucket)
    return np.array(decoding)


def encode_midi_2(decoding, track, resolution):
    offset = 0
    for row in decoding:
        timeShift = 1
        if np.sum(row[2*NUM_NOTES:]) > 0:
            timeShift = np.where(row[2*NUM_NOTES:] == 1)[0][0] + 1
        offset += int(TICK_PER_SECOND*resolution) * timeShift
        for i in np.where(row[:2*NUM_NOTES] == 1)[0]:
            if i < NUM_NOTES:
                track.append(mido.Message('note_on', note=i, velocity=DEFAULT_VELOCITY, time=offset))
            else:
                track.append(mido.Message('note_on', note=i-NUM_NOTES, velocity=0, time=offset))
            offset = 0


def decode_midi(filename, version=2, resolution=.01):
    midi = mido.MidiFile(filename)
    if version is 1:
        return decode_midi_1(midi, resolution)
    elif version is 2:
        return decode_midi_2(midi, resolution)
    return None


def encode_midi(decoding, filename, version=2, resolution=.01):
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)
    if version is 1:
        encode_midi_1(decoding, track, resolution)
    elif version is 2:
        encode_midi_2(decoding, track, resolution)
    midi.save(filename)


def main():
    encode_midi(decode_midi('testFiles/ADIG03.mid', resolution=.01),'out.mid', resolution=.01)
    encode_midi(decode_midi('testFiles/ADIG03.mid',2, resolution=.01),'out2.mid',2, resolution=.01)


if __name__== "__main__":
  main()
