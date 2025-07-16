# Flexible LULC Classification Usage Examples

## ✅ **Successfully Updated for Generic Use!**

The notebook now supports both **province-wide** and **regency-specific** analysis through a simple parameter change.

## 🚀 **Usage Examples:**

### **Option 1: Entire South Sumatra Province**
```python
# User Parameters
COUNTRY = "Indonesia"
PROVINCE = "Sumatera Selatan"
REGENCY = None  # Set to None for entire province
```

**Result:** 
- AOI: Entire Sumatera Selatan Province
- Training Points: All 39,003 points from backup asset
- Zoom Level: 8 (province-level view)
- Export File: `LULC_Sumatera_Selatan_Province_2018_17Classes.tif`

### **Option 2: Specific Regency (e.g., Banyuasin)**
```python
# User Parameters  
COUNTRY = "Indonesia"
PROVINCE = "Sumatera Selatan"
REGENCY = "Banyuasin"  # Specific regency name
```

**Result:**
- AOI: Banyuasin Regency boundaries only
- Training Points: Subset clipped to Banyuasin (~much fewer points)
- Zoom Level: 9 (regency-level view)
- Export File: `LULC_Banyuasin_Regency_2018_17Classes.tif`

### **Option 3: Different Regency**
```python
# User Parameters
COUNTRY = "Indonesia" 
PROVINCE = "Sumatera Selatan"
REGENCY = "Ogan Komering Ilir"  # Different regency
```

## 🔧 **What Changed:**

1. **Smart AOI Loading:**
   - `REGENCY = None` → Uses `get_aoi_from_gaul()` for province
   - `REGENCY = "Name"` → Uses `get_aoi_from_gaul_regency()` for regency

2. **Dynamic Naming:**
   - All outputs use `area_name` variable
   - Automatically adjusts labels and export names

3. **Appropriate Zoom Levels:**
   - Province: Zoom level 8
   - Regency: Zoom level 9

4. **Generic Messages:**
   - All print statements now use `area_name`
   - Works for any province or regency

## ✅ **Key Benefits:**

- **Parsimonious Changes:** Minimal code modifications
- **Backward Compatible:** Still works for regency analysis
- **Forward Compatible:** Easy to extend to other provinces
- **User Friendly:** Simple parameter change switches modes

## 🎯 **Perfect for Your Use Case:**

**For South Sumatra Province analysis:** Just set `REGENCY = None`
**For specific regency analysis:** Set `REGENCY = "RegencyName"`

The flexible training points system now supports both scales seamlessly! 🎉