//! Answers the first question in ADR 0006: can this machine's audio driver do echo
//! cancellation for us, using a render endpoint we choose as the reference?
//!
//!   cargo run -p grace-audio --example aec_probe
//!
//! For each active microphone it opens a shared-mode stream in the Communications category
//! (the category Windows applies voice processing to), then asks the stream for:
//! - `IAudioEffectsManager`: which effects (AEC, noise suppression, ...) are on, and whether
//!   we may toggle them;
//! - `IAcousticEchoCancellationControl`: whether we can point AEC at a reference endpoint.
//!   If this succeeds, the OS cancels everything that endpoint plays, including YouTube.
#![allow(unsafe_code)]

use windows::{
    Win32::{
        Devices::FunctionDiscovery::PKEY_Device_FriendlyName,
        Media::{Audio::*, KernelStreaming::*},
        System::Com::*,
    },
    core::{GUID, PCWSTR},
};

fn effect_name(id: &GUID) -> String {
    let known = [
        (
            AUDIO_EFFECT_TYPE_ACOUSTIC_ECHO_CANCELLATION,
            "acoustic echo cancellation",
        ),
        (AUDIO_EFFECT_TYPE_NOISE_SUPPRESSION, "noise suppression"),
        (
            AUDIO_EFFECT_TYPE_AUTOMATIC_GAIN_CONTROL,
            "automatic gain control",
        ),
        (
            AUDIO_EFFECT_TYPE_DEEP_NOISE_SUPPRESSION,
            "deep noise suppression (Voice Clarity)",
        ),
        (AUDIO_EFFECT_TYPE_BEAMFORMING, "beamforming"),
        (
            AUDIO_EFFECT_TYPE_CONSTANT_TONE_REMOVAL,
            "constant tone removal",
        ),
        (AUDIO_EFFECT_TYPE_EQUALIZER, "equalizer"),
        (AUDIO_EFFECT_TYPE_LOUDNESS_EQUALIZER, "loudness equalizer"),
        (AUDIO_EFFECT_TYPE_VIRTUAL_SURROUND, "virtual surround"),
        (AUDIO_EFFECT_TYPE_SPEAKER_PROTECTION, "speaker protection"),
        (
            AUDIO_EFFECT_TYPE_SPEAKER_COMPENSATION,
            "speaker compensation",
        ),
        (
            AUDIO_EFFECT_TYPE_DYNAMIC_RANGE_COMPRESSION,
            "dynamic range compression",
        ),
    ];
    known
        .iter()
        .find(|(g, _)| g == id)
        .map_or_else(|| format!("{id:?}"), |(_, n)| (*n).to_owned())
}

unsafe fn friendly_name(device: &IMMDevice) -> String {
    unsafe {
        device
            .OpenPropertyStore(STGM_READ)
            .and_then(|store| store.GetValue(&PKEY_Device_FriendlyName))
            .map(|v| v.to_string())
            .unwrap_or_else(|_| "<unknown>".into())
    }
}

unsafe fn probe(device: &IMMDevice, reference_id: PCWSTR) -> windows::core::Result<()> {
    unsafe {
        let client: IAudioClient2 = device.Activate(CLSCTX_ALL, None)?;
        let props = AudioClientProperties {
            cbSize: size_of::<AudioClientProperties>() as u32,
            bIsOffload: false.into(),
            eCategory: AudioCategory_Communications,
            Options: AUDCLNT_STREAMOPTIONS_NONE,
        };
        client.SetClientProperties(&props)?;
        let format = client.GetMixFormat()?;
        // WAVEFORMATEX is packed: copy it out instead of referencing its fields.
        let wf = std::ptr::read_unaligned(format);
        let (rate, channels) = (wf.nSamplesPerSec, wf.nChannels);
        println!("    mix format: {rate} Hz, {channels} ch");
        client.Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 200_000, 0, format, None)?;
        CoTaskMemFree(Some(format as _));

        match client.GetService::<IAudioEffectsManager>() {
            Ok(manager) => {
                let mut effects: *mut AUDIO_EFFECT = std::ptr::null_mut();
                let mut count = 0u32;
                manager.GetAudioEffects(&mut effects, &mut count)?;
                if count == 0 {
                    println!("    effects: none reported");
                }
                for e in std::slice::from_raw_parts(effects, count as usize) {
                    println!(
                        "    effect: {:<42} {}{}",
                        effect_name(&e.id),
                        if e.state == AUDIO_EFFECT_STATE_ON {
                            "ON"
                        } else {
                            "off"
                        },
                        if e.canSetState.as_bool() {
                            " (toggleable)"
                        } else {
                            ""
                        }
                    );
                }
                CoTaskMemFree(Some(effects as _));
            }
            Err(e) => println!("    IAudioEffectsManager: unavailable ({e})"),
        }

        match client.GetService::<IAcousticEchoCancellationControl>() {
            Ok(aec) => match aec.SetEchoCancellationRenderEndpoint(reference_id) {
                Ok(()) => println!(
                    "    AEC reference control: YES, reference set to the default speakers"
                ),
                Err(e) => println!(
                    "    AEC reference control: present, but setting the reference failed ({e})"
                ),
            },
            Err(e) => println!("    AEC reference control: NO ({e})"),
        }
        Ok(())
    }
}

fn main() -> windows::core::Result<()> {
    unsafe {
        CoInitializeEx(None, COINIT_MULTITHREADED).ok()?;
        let enumerator: IMMDeviceEnumerator =
            CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL)?;

        let speakers = enumerator.GetDefaultAudioEndpoint(eRender, eConsole)?;
        let speakers_id = speakers.GetId()?;
        println!("Default speakers: {}", friendly_name(&speakers));
        let comms_mic = enumerator.GetDefaultAudioEndpoint(eCapture, eCommunications)?;
        let comms_id = comms_mic.GetId()?.to_string()?;

        let mics = enumerator.EnumAudioEndpoints(eCapture, DEVICE_STATE_ACTIVE)?;
        for i in 0..mics.GetCount()? {
            let mic = mics.Item(i)?;
            let id = mic.GetId()?.to_string()?;
            let tag = if id == comms_id {
                " [default communications]"
            } else {
                ""
            };
            println!("\nMicrophone: {}{tag}", friendly_name(&mic));
            if let Err(e) = probe(&mic, PCWSTR(speakers_id.0)) {
                println!("    probe failed: {e}");
            }
        }
        CoTaskMemFree(Some(speakers_id.0 as _));
        Ok(())
    }
}
