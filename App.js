import React, { useState } from "react";
import {
  View,
  Text,
  Image,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  StyleSheet,
  Alert,
  SafeAreaView,
  StatusBar,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { ImageManipulator, SaveFormat } from "expo-image-manipulator";
import { API_URL, MODELS } from "./config";

// "Potato___Early_blight" -> "Early blight"
const prettyLabel = (label) =>
  label.replace(/^Potato_+/, "").replace(/_/g, " ");

export default function App() {
  const [imageUri, setImageUri] = useState(null);
  const [selectedModel, setSelectedModel] = useState(MODELS[2]); // default: Inception V1
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const resetResult = () => {
    setResult(null);
    setError(null);
  };

  const pickFromGallery = async () => {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      quality: 1,
    });
    if (!res.canceled) {
      setImageUri(res.assets[0].uri);
      resetResult();
    }
  };

  const takePhoto = async () => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      Alert.alert(
        "Permission needed",
        "Please allow camera access to take a photo.",
      );
      return;
    }
    const res = await ImagePicker.launchCameraAsync({ quality: 1 });
    if (!res.canceled) {
      setImageUri(res.assets[0].uri);
      resetResult();
    }
  };

  // Resize the photo to the size the selected model expects
  const resizeForModel = async (uri, size) => {
    const context = ImageManipulator.manipulate(uri);
    context.resize({ width: size, height: size });
    const rendered = await context.renderAsync();
    const saved = await rendered.saveAsync({
      format: SaveFormat.JPEG,
      compress: 0.95,
      base64: true, // we upload the image as base64 JSON (works on every Expo version)
    });
    return saved.base64;
  };

  const analyze = async () => {
    if (!imageUri) {
      Alert.alert("No image", "Please select or capture a leaf photo first.");
      return;
    }
    setLoading(true);
    resetResult();

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 90000); // first call loads the model

    try {
      const base64Image = await resizeForModel(imageUri, selectedModel.size);

      const response = await fetch(`${API_URL}/predict_base64`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64Image, model: selectedModel.key }),
        signal: controller.signal,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed");

      setResult(data);
    } catch (e) {
      if (e.name === "AbortError") {
        setError("The request timed out. Please try again.");
      } else if (e.message === "Network request failed") {
        setError(
          "Could not reach the server. Check API_URL in config.js and that your phone is on the same Wi-Fi.",
        );
      } else {
        setError(e.message);
      }
    } finally {
      clearTimeout(timeout);
      setLoading(false);
    }
  };

  const isHealthy = result && result.label.toLowerCase().includes("healthy");

  return (
    <SafeAreaView style={styles.safe}>
      <StatusBar barStyle="light-content" backgroundColor={GREEN} />
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Potato Leaf Disease Detector</Text>
      </View>

      <ScrollView contentContainerStyle={styles.container}>
        {/* Image preview */}
        <View style={styles.imageBox}>
          {imageUri ? (
            <Image
              source={{ uri: imageUri }}
              style={styles.image}
              resizeMode="cover"
            />
          ) : (
            <Text style={styles.placeholder}>No image selected</Text>
          )}
        </View>

        <View style={styles.row}>
          <TouchableOpacity
            style={[styles.btn, styles.btnOutline]}
            onPress={pickFromGallery}
          >
            <Text style={styles.btnOutlineText}>Choose from Gallery</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.btn, styles.btnOutline]}
            onPress={takePhoto}
          >
            <Text style={styles.btnOutlineText}>Take Photo</Text>
          </TouchableOpacity>
        </View>

        {/* Model dropdown */}
        <Text style={styles.label}>Select model</Text>
        <TouchableOpacity
          style={styles.dropdown}
          onPress={() => setDropdownOpen(!dropdownOpen)}
          activeOpacity={0.7}
        >
          <Text style={styles.dropdownText}>
            {selectedModel.name} ({selectedModel.accuracy}% accuracy)
          </Text>
          <Text style={styles.chevron}>{dropdownOpen ? "▲" : "▼"}</Text>
        </TouchableOpacity>

        {dropdownOpen && (
          <View style={styles.dropdownList}>
            {MODELS.map((m) => (
              <TouchableOpacity
                key={m.key}
                style={[
                  styles.option,
                  m.key === selectedModel.key && styles.optionActive,
                ]}
                onPress={() => {
                  setSelectedModel(m);
                  setDropdownOpen(false);
                  resetResult();
                }}
              >
                <Text style={styles.optionText}>{m.name}</Text>
                <Text style={styles.optionAcc}>{m.accuracy}%</Text>
              </TouchableOpacity>
            ))}
          </View>
        )}

        {/* Analyze */}
        <TouchableOpacity
          style={[
            styles.btn,
            styles.btnPrimary,
            (!imageUri || loading) && styles.btnDisabled,
          ]}
          onPress={analyze}
          disabled={!imageUri || loading}
        >
          {loading ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.btnPrimaryText}>Analyze Image</Text>
          )}
        </TouchableOpacity>

        {/* Error */}
        {error && (
          <View style={[styles.card, styles.errorCard]}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        {/* Result */}
        {result && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Result</Text>
            <Text
              style={[
                styles.resultLabel,
                { color: isHealthy ? GREEN : "#c62828" },
              ]}
            >
              {prettyLabel(result.label)}
            </Text>
            <Text style={styles.confidence}>
              Confidence: {result.confidence}%
            </Text>

            <View style={styles.divider} />

            {Object.entries(result.probabilities).map(([name, pct]) => (
              <View key={name} style={styles.probRow}>
                <View style={styles.probHeader}>
                  <Text style={styles.probName}>{prettyLabel(name)}</Text>
                  <Text style={styles.probName}>{pct}%</Text>
                </View>
                <View style={styles.barBg}>
                  <View
                    style={[
                      styles.barFill,
                      { width: `${Math.min(pct, 100)}%` },
                    ]}
                  />
                </View>
              </View>
            ))}

            <Text style={styles.meta}>
              Model: {selectedModel.name} · Input resized to {result.input_size}
              ×{result.input_size}
            </Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const GREEN = "#2e7d32";

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#f4f6f4" },
  header: {
    backgroundColor: GREEN,
    paddingVertical: 16,
    paddingHorizontal: 20,
  },
  headerTitle: { color: "#fff", fontSize: 18, fontWeight: "600" },
  container: { padding: 16, paddingBottom: 40 },

  imageBox: {
    height: 260,
    borderRadius: 12,
    backgroundColor: "#e3e8e3",
    justifyContent: "center",
    alignItems: "center",
    overflow: "hidden",
    marginBottom: 12,
  },
  image: { width: "100%", height: "100%" },
  placeholder: { color: "#7a857a", fontSize: 15 },

  row: { flexDirection: "row", gap: 10, marginBottom: 16 },
  btn: {
    borderRadius: 8,
    paddingVertical: 13,
    alignItems: "center",
    justifyContent: "center",
  },
  btnOutline: {
    flex: 1,
    borderWidth: 1,
    borderColor: GREEN,
    backgroundColor: "#fff",
  },
  btnOutlineText: { color: GREEN, fontWeight: "600" },
  btnPrimary: { backgroundColor: GREEN, marginTop: 16 },
  btnPrimaryText: { color: "#fff", fontWeight: "600", fontSize: 16 },
  btnDisabled: { opacity: 0.5 },

  label: { fontSize: 14, color: "#444", marginBottom: 6 },
  dropdown: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: "#fff",
    borderWidth: 1,
    borderColor: "#c9d1c9",
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 13,
  },
  dropdownText: { fontSize: 15, color: "#222" },
  chevron: { color: "#666", fontSize: 12 },
  dropdownList: {
    backgroundColor: "#fff",
    borderWidth: 1,
    borderColor: "#c9d1c9",
    borderRadius: 8,
    marginTop: 4,
    overflow: "hidden",
  },
  option: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 14,
    paddingVertical: 13,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: "#ddd",
  },
  optionActive: { backgroundColor: "#e8f2e8" },
  optionText: { fontSize: 15, color: "#222" },
  optionAcc: { fontSize: 14, color: "#666" },

  card: {
    backgroundColor: "#fff",
    borderRadius: 12,
    padding: 16,
    marginTop: 18,
    borderWidth: 1,
    borderColor: "#e0e5e0",
  },
  cardTitle: {
    fontSize: 13,
    color: "#777",
    textTransform: "uppercase",
    marginBottom: 6,
  },
  resultLabel: { fontSize: 24, fontWeight: "700" },
  confidence: { fontSize: 15, color: "#444", marginTop: 4 },
  divider: { height: 1, backgroundColor: "#eee", marginVertical: 14 },
  probRow: { marginBottom: 10 },
  probHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 4,
  },
  probName: { fontSize: 14, color: "#333" },
  barBg: {
    height: 8,
    backgroundColor: "#e6ebe6",
    borderRadius: 4,
    overflow: "hidden",
  },
  barFill: { height: 8, backgroundColor: GREEN },
  meta: { fontSize: 12, color: "#888", marginTop: 8 },

  errorCard: { backgroundColor: "#fdecea", borderColor: "#f5c6c2" },
  errorText: { color: "#b71c1c", fontSize: 14 },
});
