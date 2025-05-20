// Scrape recent(2024 'till 5/10/2025) Smartphones from GSM Arena
const list = [
  "samsung_galaxy_a56-13603",
  "xiaomi_redmi_turbo_4_pro-13617",
  "samsung_galaxy_s25_ultra-13322",
  "oneplus_13t_5g-13791",
  "xiaomi_poco_x7_pro-13582",
  "nothing_cmf_phone_2_pro_5g-13821",
  "samsung_galaxy_f56_5g-13855",
  "apple_iphone_16_pro_max-13123",
  "motorola_edge_60_pro-13815",
  "samsung_galaxy_a36-13497",
  "vivo_y300_gt_5g-13857",
  "tecno_camon_40_pro_5g-13695",
  "samsung_galaxy_a55-12824",
  "samsung_galaxy_s24_ultra-12771",
  "infinix_note_50_pro+-13746",
  "samsung_galaxy_s24_fe-13262",
  "vivo_x200_ultra-13762",
  "xiaomi_redmi_note_14_pro_5g_(global)-13613",
  "samsung_galaxy_s25-13610",
  "samsung_galaxy_a16_5g-13346",
  "xiaomi_redmi_14c-13291",
  "xiaomi_redmi_note_14_4g_(global)-13616",
  "samsung_galaxy_a06-13265",
  "infinix_note_50_pro_4g-13701",
  "samsung_galaxy_a26-13679",
  "apple_iphone_16-13317",
  "samsung_galaxy_s24-12773",
  "xiaomi_poco_f7_pro-13724",
  "samsung_galaxy_a35-12705",
  "samsung_galaxy_a16-13383",
  "google_pixel_9a-13478",
  "xiaomi_15_ultra-13657",
  "xiaomi_redmi_note_13-12776",
  "xiaomi_redmi_note_14_pro+_5g_(global)-13612",
  "apple_iphone_16_pro-13315",
  "vivo_v50-13656",
  "honor_x9c-13490",
  "xiaomi_redmi_a5_4g-13737",
  "motorola_edge_60_fusion-13752",
  "oneplus_13-13477",
  "xiaomi_redmi_note_14_5g_(global)-13615",
  "xiaomi_poco_f7_ultra-13723",
  "itel_city_100-13853",
  "tecno_camon_40-13697",
  "google_pixel_9_pro-13218",
  "vivo_iqoo_z10_turbo_pro_5g-13800",
  "xiaomi_15-13472",
  "oppo_a5_pro-13675",
  "samsung_galaxy_m56_5g-13801",
  "nothing_phone_(3a)-13672",
  "motorola_edge_60-13779",
  "sony_xperia_1_vi-13003",
  "infinix_hot_50-13305",
  "infinix_smart_9-13374",
  "infinix_hot_50_pro+_4g-13408",
  "google_pixel_9_pro_xl-13217",
  "apple_iphone_16e-13395",
  "infinix_gt_20_pro-12962",
  "realme_14-13729",
  "xiaomi_poco_x6_pro-12717",
  "nothing_phone_(3a)_pro-13649",
  "oneplus_13s_5g-13818",
  "vivo_v50_lite-13745",
  "xiaomi_redmi_turbo_4-13598",
  "motorola_moto_g85-13144",
  "vivo_t4_5g-13781",
  "infinix_hot_50i-13375",
  "xiaomi_redmi_13-13102",
  "oneplus_13r-13548",
  "vivo_x200_pro-13410",
];

// START - bring them one by one

import { catalog } from "gsmarena-api";

const devices = [];

for (const key of list) {
  try {
    const device = await catalog.getDevice(key);

    devices.push(device);
  } catch {
    console.log("err");
  }
}

// END - bring them one by one

async function postEndpointData(deviceId) {
  const url = `https://script.google.com/macros/s/AKfycbxNu27V2Y2LuKUIQMK8lX1y0joB6YmG6hUwB1fNeVbgzEh22TcDGrOak03Fk3uBHmz-/exec`;
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: {
        route: "device-detail",
        key: deviceId,
      },
    });
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error posting data to", url, ":", error);
    throw error;
  }
}
