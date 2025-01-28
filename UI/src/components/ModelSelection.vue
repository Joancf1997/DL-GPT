<template>
    <div class="card">
        <Menubar :model="items" />
    </div>
</template>

<script setup>
import { ref } from 'vue';
import Menubar from 'primevue/menubar';


const items = ref([
    {
        label: 'Classification',
        icon: 'pi pi-microchip-ai'
    },
    {
        label: 'Assistant',
        icon: 'pi pi-microchip-ai'
    }
]);
</script>

<style scoped>
.card{
    width: 100%;
    height: 55px;
}
</style>
